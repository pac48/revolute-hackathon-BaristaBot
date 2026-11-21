// Records the topics a policy is trained from into an MCAP rosbag, without losing messages.
//
//   bazel run //src/data_recorder:record -- [-o DIR] [--use-sim-time] [--duration S] ...
//
// What: every controller's state (control_msgs/JointTrajectoryControllerState: the commanded targets are
// its reference), every sensor_msgs Image, CameraInfo and CompressedImage topic (found as they appear;
// a CompressedImage is skipped when its raw Image is recorded), and --topics (default /joint_states,
// /tf, /tf_static). Images are compressed here: color to JPEG (--jpeg-quality) on <topic>/compressed,
// depth (16UC1, or 32FC1 in meters, stored as 16UC1 millimeters like the real Gemini's) losslessly
// (--depth-format png|rvl) on <topic>/compressedDepth, in image_transport's formats so the usual tools
// decode them. --raw records them as they come.
//
// Without loss: the subscription callbacks only queue the serialized message (they never wait for
// encoding or the disk); --encode-threads workers compress and write. The queue holds up to --buffer-mb,
// so encoding or disk stalls (a file split, a slow flush) only delay the writes. A message is lost only
// if the queue is full (the disk can't keep up for long), or before it reaches us: in DDS, e.g. when a
// publisher with a short history overwrites a sample we have not received yet. Both are counted, the
// latter from the gaps in each publisher's sequence numbers. The stats go to stdout every --stats-period
// seconds and at the end; the exit code is 2 if anything was lost, 3 if no topic was found. Ctrl-C (or
// --duration) stops subscribing, then writes everything queued before closing the bag.
//
// Episodes (--episodic): nothing is recorded until a client calls ~/start (std_srvs/Trigger), e.g. when a task
// episode begins, rather than its reset. Each episode is its own bag, DIR/episode_NNNN: ~/stop writes what is
// queued, finalizes that bag and opens the next one. ~/start while an episode is recording discards it (its bag
// is deleted and the episode starts over under the same number): a client that never stops a bad or mistaken
// episode throws it out by starting the next. Both reply with the bag's path. At exit an episode still
// recording is discarded too (only ~/stop keeps one), and the opened, never started bag removed. Between episodes the subscriptions
// stay (so publishers' sequence numbers are still followed) but nothing is queued.
//
// Voice: "Start demonstration" when recording starts (each ~/start with --episodic, else at launch) and "End
// demonstration" when it stops (~/stop, else at Ctrl-C/--duration), from sounds/ through paplay (or aplay) on this
// machine's default audio output. --no-voice turns it off; without a player it only logs a warning.
//
// Disk: compressed, the sim's two cameras come to ~4 MB/s. Raw (--raw) they are ~340 MB/s, more than a
// budget NVMe drive sustains once the page cache is full (the dev PC's: ~140 MB/s after ~80 s), and the
// queue then fills. --test-write-stall blocks every write for a while, to check that the queue absorbs it.
#include <cxxopts.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rules_cc/cc/runfiles/runfiles.h>
#include <spawn.h>
#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

extern char ** environ;

#include "compressed_depth_image_transport/codec.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/writer.hpp"
#include "rosbag2_storage/storage_options.hpp"
#include "rosbag2_storage/topic_metadata.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace
{
constexpr const char * kImage = "sensor_msgs/msg/Image";
constexpr const char * kCompressedImage = "sensor_msgs/msg/CompressedImage";
constexpr const char * kCameraInfo = "sensor_msgs/msg/CameraInfo";
constexpr const char * kControllerState = "control_msgs/msg/JointTrajectoryControllerState";

struct Options
{
  std::string output;
  bool use_sim_time = false;
  double duration = 0.0;
  bool raw = false;
  int jpeg_quality = 90;
  std::string depth_format = "png";
  int png_level = 1;
  int encode_threads = 8;
  size_t buffer_bytes = 0;
  int queue_depth = 100;
  std::vector<std::string> topics;
  double stats_period = 2.0;
  uint64_t max_file_bytes = 0;
  bool episodic = false;
  bool voice = true;
  double test_stall = 0.0, test_stall_period = 10.0;
};

/// One recorded topic: what comes in and its counters.
struct Topic
{
  std::string name, type;
  bool image = false;  // an Image to compress
  std::vector<rclcpp::QoS> offered_qos;
  rclcpp::GenericSubscription::SharedPtr subscription;
  std::unordered_map<std::string, uint64_t> last_sequence;  // per publisher (gid); executor thread only
  std::atomic<uint64_t> received{0}, written{0}, bytes_in{0}, bytes_out{0}, lost_dds{0}, lost_queue{0},
    failed{0};
};

struct Item
{
  Topic * topic;
  std::shared_ptr<rclcpp::SerializedMessage> message;
  int64_t recv_ns;
  uint32_t sequence;  // the publisher's (truncated), stored in the bag for bag_check; 0: unknown
  uint64_t episode;  // the recording it belongs to (Recorder::episode_)
};

/// Plays the start/end demonstration clips without blocking the caller: each in a child process (paplay, else
/// aplay) that a detached thread reaps.
class Voice
{
public:
  explicit Voice(bool enabled, rclcpp::Logger logger) : logger_(logger)
  {
    if (!enabled) return;
    std::error_code ec;
    const std::string exe = std::filesystem::read_symlink("/proc/self/exe", ec).string();
    std::string error;
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::Create(exe, BAZEL_CURRENT_REPOSITORY, &error));
    if (!runfiles)
    {
      RCLCPP_WARN(logger_, "No voice announcements: cannot locate runfiles (%s)", error.c_str());
      return;
    }
    start_ = runfiles->Rlocation("_main/src/data_recorder/sounds/start_demonstration.wav");
    end_ = runfiles->Rlocation("_main/src/data_recorder/sounds/end_demonstration.wav");
  }

  void start() { play(start_); }
  void end() { play(end_); }

private:
  void play(const std::string & wav)
  {
    if (wav.empty()) return;
    std::thread([wav, logger = logger_, warned = &warned_] {
      for (const char * player : {"paplay", "aplay"})
      {
        std::vector<std::string> args{player, wav};
        if (std::strcmp(player, "aplay") == 0) args.insert(args.begin() + 1, "-q");
        std::vector<char *> argv;
        for (auto & arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        pid_t pid;
        if (posix_spawnp(&pid, player, nullptr, nullptr, argv.data(), environ) != 0) continue;  // not installed
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return;
      }
      if (!warned->exchange(true))
        RCLCPP_WARN(logger, "Cannot play %s: no working paplay or aplay (no audio output?)", wav.c_str());
    }).detach();
  }

  rclcpp::Logger logger_;
  std::string start_, end_;
  std::atomic<bool> warned_{false};
};

std::string default_output()
{
  const char * cwd = std::getenv("BUILD_WORKING_DIRECTORY");  // bazel run: where it was run from
  char stamp[64];
  const std::time_t now = std::time(nullptr);
  std::strftime(stamp, sizeof(stamp), "rosbag2_%Y_%m_%d-%H_%M_%S", std::localtime(&now));
  return (std::filesystem::path(cwd ? cwd : ".") / "recordings" / stamp).string();
}

bool is_depth(const std::string & encoding)
{
  return encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
         encoding == sensor_msgs::image_encodings::TYPE_32FC1 || encoding == sensor_msgs::image_encodings::MONO16;
}

class Recorder : public rclcpp::Node
{
public:
  explicit Recorder(const Options & options)
  : rclcpp::Node(
      "data_recorder",
      rclcpp::NodeOptions().parameter_overrides({rclcpp::Parameter("use_sim_time", options.use_sim_time)})),
    options_(options),
    voice_(options.voice, get_logger())
  {
    if (options_.episodic)
    {
      std::filesystem::create_directories(options_.output);
      open_bag(episode_path());
      // Their own group (a second executor thread), so finalizing a bag never holds up receiving. The node only
      // keeps a weak reference to it.
      service_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      using Trigger = std_srvs::srv::Trigger;
      start_service_ = create_service<Trigger>(
        "~/start", [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
          start_episode(*response);
        }, rclcpp::ServicesQoS(), service_group_);
      stop_service_ = create_service<Trigger>(
        "~/stop", [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
          stop_episode(*response);
        }, rclcpp::ServicesQoS(), service_group_);
      RCLCPP_INFO(get_logger(), "Episodes into %s: waiting for ~/start", options_.output.c_str());
    }
    else
    {
      open_bag(options_.output);
      recording_ = true;
      RCLCPP_INFO(get_logger(), "Recording to %s", options_.output.c_str());
      voice_.start();
    }

    for (int i = 0; i < options_.encode_threads; ++i) workers_.emplace_back([this] { work(); });
    discover();
    discovery_timer_ = create_wall_timer(std::chrono::seconds(1), [this] { discover(); });
    stats_timer_ = create_wall_timer(
      std::chrono::duration<double>(options_.stats_period), [this] { print_stats(false); });
    start_ = std::chrono::steady_clock::now();
  }

  /// Stop subscribing, write what is queued, close the bag. Returns the number of messages lost.
  uint64_t finish()
  {
    discovery_timer_.reset();
    stats_timer_.reset();
    for (auto & topic : topics_) topic->subscription.reset();
    if (!options_.episodic) voice_.end();
    {
      std::unique_lock lock(queue_mutex_);
      RCLCPP_INFO(get_logger(), "Stopping: writing the %zu queued messages (%.0f MiB)", queue_.size(), queue_bytes_ / 1048576.0);
      closing_ = true;
      closing_flag_ = true;
    }
    queue_cv_.notify_all();
    for (auto & worker : workers_) worker.join();
    writer_->close();
    if (options_.episodic)
    {
      // Only ~/stop keeps an episode: one still recording is discarded, like the bag opened for the next.
      if (recording_) RCLCPP_WARN(get_logger(), "Episode %d was not stopped: discarded", episode_number_);
      std::filesystem::remove_all(bag_path_);
    }
    print_stats(true);
    uint64_t lost = 0;
    for (const auto & topic : topics_) lost += topic->lost_dds + topic->lost_queue;
    return lost;
  }

  bool expired() const
  {
    return options_.duration > 0 &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count() >= options_.duration;
  }

  size_t num_topics() const { return topics_.size(); }

private:
  bool wanted(const std::string & name, const std::string & type, const std::map<std::string, std::vector<std::string>> & all) const
  {
    if (std::find(options_.topics.begin(), options_.topics.end(), name) != options_.topics.end()) return true;
    // Hidden topics (a token starting with '_'), e.g. rmw_fastrtps's <topic>/_buf_cpu channel of a buffer-aware
    // publisher: the same messages as <topic>, and subscribing to it changes how the publisher delivers them.
    if (name.find("/_") != std::string::npos) return false;
    if (type == kImage || type == kCameraInfo || type == kControllerState) return true;
    if (type == kCompressedImage)
    {
      // Recorded from its raw image instead, if that is there (image_transport's <base>/<transport>).
      const auto slash = name.rfind('/');
      const auto base = all.find(name.substr(0, slash));
      return slash == std::string::npos || base == all.end() ||
             std::find(base->second.begin(), base->second.end(), kImage) == base->second.end();
    }
    return false;
  }

  void discover()
  {
    const auto all = get_topic_names_and_types();
    for (const auto & [name, types] : all)
    {
      if (types.size() != 1 || subscribed_.count(name) || !wanted(name, types[0], all)) continue;
      const auto publishers = get_publishers_info_by_topic(name);
      if (publishers.empty()) continue;  // only subscribers so far: their QoS is not the publishers'
      // Reliable (and transient local, for latched topics like /tf_static) only if every publisher offers it,
      // or they would not match.
      bool reliable = true, transient_local = true;
      auto topic = std::make_unique<Topic>();
      for (const auto & publisher : publishers)
      {
        const auto & qos = publisher.qos_profile();
        reliable &= qos.reliability() == rclcpp::ReliabilityPolicy::Reliable;
        transient_local &= qos.durability() == rclcpp::DurabilityPolicy::TransientLocal;
        topic->offered_qos.push_back(qos);
      }
      rclcpp::QoS qos(rclcpp::KeepLast(static_cast<size_t>(options_.queue_depth)));
      qos.reliability(reliable ? rclcpp::ReliabilityPolicy::Reliable : rclcpp::ReliabilityPolicy::BestEffort);
      qos.durability(transient_local ? rclcpp::DurabilityPolicy::TransientLocal : rclcpp::DurabilityPolicy::Volatile);

      topic->name = name;
      topic->type = types[0];
      topic->image = types[0] == kImage && !options_.raw;
      Topic * raw = topic.get();
      try
      {
        topic->subscription = create_generic_subscription(
          name, types[0], qos,
          [this, raw](std::shared_ptr<const rclcpp::SerializedMessage> message, const rclcpp::MessageInfo & info) {
            receive(*raw, message, info);
          });
      }
      catch (const std::exception & e)
      {
        RCLCPP_ERROR(get_logger(), "Cannot record %s (%s): %s", name.c_str(), types[0].c_str(), e.what());
        subscribed_.insert(name);  // don't retry every second
        continue;
      }
      RCLCPP_INFO(
        get_logger(), "Recording %s (%s, %s%s)", name.c_str(), types[0].c_str(), reliable ? "reliable" : "best effort",
        transient_local ? ", transient local" : "");
      subscribed_.insert(name);
      topics_.push_back(std::move(topic));
    }
  }

  // Executor thread: count, check the sequence, queue. Never waits.
  void receive(Topic & topic, std::shared_ptr<const rclcpp::SerializedMessage> message, const rclcpp::MessageInfo & info)
  {
    const int64_t recv_ns = now().nanoseconds();
    const auto & rmw = info.get_rmw_message_info();
    ++topic.received;
    const size_t size = message->size();
    topic.bytes_in += size;
    if (rmw.publication_sequence_number != RMW_MESSAGE_INFO_SEQUENCE_NUMBER_UNSUPPORTED)
    {
      const std::string gid(reinterpret_cast<const char *>(rmw.publisher_gid.data), sizeof(rmw.publisher_gid.data));
      auto [it, first] = topic.last_sequence.try_emplace(gid, rmw.publication_sequence_number);
      if (!first)
      {
        if (rmw.publication_sequence_number > it->second + 1)
        {
          const uint64_t gap = rmw.publication_sequence_number - it->second - 1;
          topic.lost_dds += gap;
          RCLCPP_ERROR_THROTTLE(
            get_logger(), *get_clock(), 1000, "%s: %" PRIu64 " messages lost before they reached the recorder",
            topic.name.c_str(), gap);
        }
        it->second = std::max(it->second, rmw.publication_sequence_number);
      }
    }
    {
      std::lock_guard lock(queue_mutex_);
      if (!recording_) return;  // between episodes
      if (queue_bytes_ + size > options_.buffer_bytes)
      {
        ++topic.lost_queue;
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000, "Queue full (%zu MB): dropped a message of %s; the disk or the encoding can't keep up",
          options_.buffer_bytes >> 20, topic.name.c_str());
        return;
      }
      // The subscription hands us a const message it no longer uses: keep it without copying.
      const uint32_t sequence = rmw.publication_sequence_number == RMW_MESSAGE_INFO_SEQUENCE_NUMBER_UNSUPPORTED
                                  ? 0
                                  : static_cast<uint32_t>(rmw.publication_sequence_number);
      queue_.push_back({&topic, std::const_pointer_cast<rclcpp::SerializedMessage>(message), recv_ns, sequence, episode_});
      queue_bytes_ += size;
      ++pending_;
      peak_queue_bytes_ = std::max(peak_queue_bytes_, queue_bytes_);
    }
    queue_cv_.notify_one();
  }

  // Worker: compress (images) and write, until closing and the queue is empty.
  void work()
  {
    for (;;)
    {
      Item item;
      bool drop = false;  // its episode was discarded
      {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return closing_ || !queue_.empty(); });
        if (queue_.empty()) return;  // closing
        item = std::move(queue_.front());
        queue_.pop_front();
        queue_bytes_ -= item.message->size();
        drop = item.episode == discarded_episode_;
      }
      if (!drop) process(item);
      {
        std::lock_guard lock(queue_mutex_);
        --pending_;
      }
      done_cv_.notify_all();
    }
  }

  // Compress (images) and write one item.
  void process(const Item & item)
  {
    {
      Topic & topic = *item.topic;
      std::string out_name = topic.name, out_type = topic.type;
      std::shared_ptr<rclcpp::SerializedMessage> out = item.message;
      if (topic.image)
      {
        out = compress(topic, *item.message, out_name);
        out_type = kCompressedImage;
        if (!out)
        {
          ++topic.failed;
          return;
        }
      }
      {
        std::lock_guard lock(write_mutex_);
        stall_if_due();
        if (created_.insert(out_name).second)
        {
          rosbag2_storage::TopicMetadata metadata;
          metadata.name = out_name;
          metadata.type = out_type;
          metadata.serialization_format = "cdr";
          metadata.offered_qos_profiles = topic.offered_qos;
          writer_->create_topic(metadata);
        }
        try
        {
          writer_->write(out, out_name, out_type, item.recv_ns, item.recv_ns, item.sequence);
        }
        catch (const std::exception & e)
        {
          ++topic.failed;
          RCLCPP_ERROR(get_logger(), "Writing %s failed: %s", out_name.c_str(), e.what());
          return;
        }
      }
      ++topic.written;
      ++episode_written_;
      topic.bytes_out += out->size();
    }
  }

  std::string episode_path() const
  {
    char name[32];
    std::snprintf(name, sizeof(name), "episode_%04d", episode_number_);
    return (std::filesystem::path(options_.output) / name).string();
  }

  // A new bag at `path` for the writers (the previous one closed).
  void open_bag(const std::string & path)
  {
    std::lock_guard lock(write_mutex_);
    rosbag2_storage::StorageOptions storage;
    storage.uri = path;
    storage.storage_id = "mcap";
    storage.max_cache_size = 0;  // write straight through: our queue is the buffer
    storage.max_bagfile_size = options_.max_file_bytes;
    writer_ = std::make_unique<rosbag2_cpp::Writer>();
    writer_->open(storage);
    created_.clear();
    bag_path_ = path;
  }

  // Stop queueing the current episode and wait until the workers are done with what it queued (written, or
  // dropped if `discard`), so its bag can be closed.
  void drain_episode(bool discard)
  {
    std::unique_lock lock(queue_mutex_);
    recording_ = false;
    if (discard) discarded_episode_ = episode_;
    done_cv_.wait(lock, [this] { return pending_ == 0; });
  }

  void close_bag()
  {
    std::lock_guard lock(write_mutex_);
    writer_->close();
  }

  // ~/start: begin an episode, discarding the one recording if there is one.
  void start_episode(std_srvs::srv::Trigger::Response & response)
  {
    if (recording_)
    {
      drain_episode(true);
      close_bag();
      std::filesystem::remove_all(bag_path_);
      RCLCPP_WARN(get_logger(), "Episode %d discarded (started again)", episode_number_);
      open_bag(bag_path_);
    }
    {
      std::lock_guard lock(queue_mutex_);
      ++episode_;
      recording_ = true;
    }
    episode_written_ = 0;
    episode_start_ = std::chrono::steady_clock::now();
    voice_.start();
    RCLCPP_INFO(get_logger(), "Episode %d recording to %s", episode_number_, bag_path_.c_str());
    response.success = true;
    response.message = bag_path_;
  }

  // ~/stop: finalize the episode's bag and open the next one.
  void stop_episode(std_srvs::srv::Trigger::Response & response)
  {
    if (!recording_)
    {
      response.success = false;
      response.message = "not recording: call ~/start first";
      return;
    }
    voice_.end();  // now, not after the bag is written
    drain_episode(false);
    close_bag();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - episode_start_).count();
    RCLCPP_INFO(
      get_logger(), "Episode %d finalized: %" PRIu64 " messages, %.1f s, %s", episode_number_, episode_written_.load(),
      seconds, bag_path_.c_str());
    response.success = true;
    response.message = bag_path_;
    ++episode_number_;
    open_bag(episode_path());
  }

  // --test-write-stall: block every writer (we hold write_mutex_) as a hung disk write would.
  void stall_if_due()
  {
    if (options_.test_stall <= 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (next_stall_ == std::chrono::steady_clock::time_point{}) next_stall_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(options_.test_stall_period));
    if (now < next_stall_ || closing_flag_) return;
    RCLCPP_WARN(get_logger(), "Test: stalling the writes for %.1f s", options_.test_stall);
    std::this_thread::sleep_for(std::chrono::duration<double>(options_.test_stall));
    next_stall_ = std::chrono::steady_clock::now() +
                  std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(options_.test_stall_period));
  }

  /// The image as a serialized CompressedImage on <topic>/compressed (color) or <topic>/compressedDepth
  /// (depth); nullptr (and a log) if it can't be.
  std::shared_ptr<rclcpp::SerializedMessage> compress(const Topic & topic, const rclcpp::SerializedMessage & serialized, std::string & out_name)
  {
    sensor_msgs::msg::Image image;
    image_serialization_.deserialize_message(&serialized, &image);
    namespace enc = sensor_msgs::image_encodings;
    sensor_msgs::msg::CompressedImage compressed;
    if (is_depth(image.encoding))
    {
      out_name = topic.name + "/compressedDepth";
      const sensor_msgs::msg::Image * depth = &image;
      sensor_msgs::msg::Image millimeters;
      if (image.encoding == enc::TYPE_32FC1)
      {
        // Meters to millimeters, as the real camera gives them; invalid (NaN, inf, <= 0) or beyond 65.535 m: 0.
        millimeters.header = image.header;
        millimeters.height = image.height;
        millimeters.width = image.width;
        millimeters.encoding = enc::TYPE_16UC1;
        millimeters.is_bigendian = image.is_bigendian;
        millimeters.step = image.width * 2;
        millimeters.data.resize(static_cast<size_t>(millimeters.step) * image.height);
        for (uint32_t row = 0; row < image.height; ++row)
        {
          const auto * in = reinterpret_cast<const float *>(&image.data[static_cast<size_t>(row) * image.step]);
          auto * out = reinterpret_cast<uint16_t *>(&millimeters.data[static_cast<size_t>(row) * millimeters.step]);
          for (uint32_t col = 0; col < image.width; ++col)
          {
            const float mm = in[col] * 1000.0f;
            out[col] = std::isfinite(mm) && mm > 0.0f && mm < 65535.5f ? static_cast<uint16_t>(std::lround(mm)) : 0;
          }
        }
        depth = &millimeters;
      }
      auto encoded = compressed_depth_image_transport::encodeCompressedDepthImage(
        *depth, options_.depth_format, 10.0, 100.0, options_.png_level);
      if (!encoded)
      {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s: cannot compress depth (%s)", topic.name.c_str(), image.encoding.c_str());
        return nullptr;
      }
      compressed = std::move(*encoded);
    }
    else
    {
      out_name = topic.name + "/compressed";
      int type = 0, to_bgr = -1;
      std::string target = "bgr8";
      if (image.encoding == enc::RGB8) type = CV_8UC3, to_bgr = cv::COLOR_RGB2BGR;
      else if (image.encoding == enc::BGR8) type = CV_8UC3;
      else if (image.encoding == enc::RGBA8) type = CV_8UC4, to_bgr = cv::COLOR_RGBA2BGR;
      else if (image.encoding == enc::BGRA8) type = CV_8UC4, to_bgr = cv::COLOR_BGRA2BGR;
      else if (image.encoding == enc::MONO8) type = CV_8UC1, target = "mono8";
      else
      {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s: cannot compress encoding %s", topic.name.c_str(), image.encoding.c_str());
        return nullptr;
      }
      const cv::Mat in(static_cast<int>(image.height), static_cast<int>(image.width), type, image.data.data(), image.step);
      cv::Mat bgr = in;
      if (to_bgr >= 0) cv::cvtColor(in, bgr, to_bgr);
      std::vector<uchar> jpeg;
      if (!cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, options_.jpeg_quality}))
      {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "%s: JPEG encoding failed", topic.name.c_str());
        return nullptr;
      }
      compressed.header = image.header;
      compressed.format = image.encoding + "; jpeg compressed " + target;  // compressed_image_transport's
      compressed.data = std::move(jpeg);
    }
    auto out = std::make_shared<rclcpp::SerializedMessage>();
    compressed_serialization_.serialize_message(&compressed, out.get());
    return out;
  }

  void print_stats(bool final)
  {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    const double period = final ? elapsed : elapsed - last_stats_elapsed_;
    size_t queued, peak;
    {
      std::lock_guard lock(queue_mutex_);
      queued = queue_bytes_;
      peak = peak_queue_bytes_;
    }
    std::printf(
      "%s %.0f s, queue %.0f MiB (peak %.0f of %zu MiB)\n", final ? "== Recorded" : "--", elapsed, queued / 1048576.0,
      peak / 1048576.0, options_.buffer_bytes >> 20);
    std::printf("  %-48s %8s %8s %9s %9s %6s %6s %6s\n", "topic", "msgs", "Hz", "in MB/s", "out MB/s", "lost", "drop", "fail");
    uint64_t lost = 0;
    for (const auto & topic : topics_)
    {
      auto & last = last_stats_[topic.get()];
      const uint64_t received = topic->received, in = topic->bytes_in, out = topic->bytes_out;
      std::printf(
        "  %-48s %8" PRIu64 " %8.1f %9.1f %9.1f %6" PRIu64 " %6" PRIu64 " %6" PRIu64 "\n", topic->name.c_str(), received,
        (final ? received : received - last[0]) / period, (final ? in : in - last[1]) / period / 1e6,
        (final ? out : out - last[2]) / period / 1e6, topic->lost_dds.load(), topic->lost_queue.load(), topic->failed.load());
      last = {received, in, out};
      lost += topic->lost_dds + topic->lost_queue;
    }
    if (final)
    {
      std::printf(
        "%s: %s\n", lost ? "LOST MESSAGES" : "No messages lost", options_.output.c_str());
    }
    std::fflush(stdout);
    last_stats_elapsed_ = elapsed;
  }

  Options options_;
  Voice voice_;
  std::unique_ptr<rosbag2_cpp::Writer> writer_;
  std::mutex write_mutex_;
  std::set<std::string> created_;  // bag topics (write_mutex_)
  std::chrono::steady_clock::time_point next_stall_{};  // write_mutex_
  std::atomic<bool> closing_flag_{false};

  std::vector<std::unique_ptr<Topic>> topics_;
  std::set<std::string> subscribed_;
  rclcpp::TimerBase::SharedPtr discovery_timer_, stats_timer_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_, done_cv_;
  // What is queued belongs to episode_ while recording_; pending_ counts its items queued or in a worker.
  std::atomic<bool> recording_{false};  // set under queue_mutex_
  uint64_t episode_ = 0, discarded_episode_ = UINT64_MAX;
  size_t pending_ = 0;
  // --episodic: the bag being written, its number, and the services.
  std::string bag_path_;
  int episode_number_ = 1;
  std::atomic<uint64_t> episode_written_{0};
  std::chrono::steady_clock::time_point episode_start_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_service_, stop_service_;
  std::deque<Item> queue_;
  size_t queue_bytes_ = 0, peak_queue_bytes_ = 0;
  bool closing_ = false;
  std::vector<std::thread> workers_;

  rclcpp::Serialization<sensor_msgs::msg::Image> image_serialization_;
  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> compressed_serialization_;

  std::chrono::steady_clock::time_point start_;
  double last_stats_elapsed_ = 0.0;
  std::map<const Topic *, std::array<uint64_t, 3>> last_stats_;
};

}  // namespace

int main(int argc, char ** argv)
{
  cxxopts::Options cli("record", "Record controller states and camera images to an MCAP rosbag without loss.");
  cli.add_options()
    ("o,output", "Bag directory (default: recordings/rosbag2_<time> where it is run from)", cxxopts::value<std::string>())
    ("use-sim-time", "Stamp with /clock (the sim's time)", cxxopts::value<bool>()->default_value("false"))
    ("duration", "Stop after this many seconds (0: at Ctrl-C)", cxxopts::value<double>()->default_value("0"))
    ("raw", "Record images uncompressed", cxxopts::value<bool>()->default_value("false"))
    ("jpeg-quality", "Color JPEG quality (1-100)", cxxopts::value<int>()->default_value("90"))
    ("depth-format", "Lossless depth codec: png or rvl", cxxopts::value<std::string>()->default_value("png"))
    ("png-level", "PNG compression level for depth (0-9: smaller but slower)", cxxopts::value<int>()->default_value("1"))
    ("encode-threads", "Threads compressing and writing", cxxopts::value<int>()->default_value("8"))
    ("buffer-mb", "Queue between the subscriptions and the writers (MB)", cxxopts::value<size_t>()->default_value("4096"))
    ("queue-depth", "Subscription history depth (messages)", cxxopts::value<int>()->default_value("100"))
    ("topics", "Other topics to record", cxxopts::value<std::vector<std::string>>()->default_value("/joint_states,/tf,/tf_static"))
    ("stats-period", "Seconds between stats lines", cxxopts::value<double>()->default_value("2"))
    ("max-file-mb", "Split the bag into files of about this size (0: one file)", cxxopts::value<uint64_t>()->default_value("0"))
    ("episodic", "Record episodes between ~/start and ~/stop calls, one bag each in the output directory",
     cxxopts::value<bool>()->default_value("false"))
    ("no-voice", "Don't say \"Start/End demonstration\" when recording starts and stops",
     cxxopts::value<bool>()->default_value("false"))
    ("test-write-stall", "Testing: block all writes this many seconds ...", cxxopts::value<double>()->default_value("0"))
    ("test-write-stall-period", "... every this many seconds", cxxopts::value<double>()->default_value("10"))
    ("h,help", "Show this help");
  cxxopts::ParseResult args;
  try
  {
    args = cli.parse(argc, argv);
  }
  catch (const std::exception & e)
  {
    std::fprintf(stderr, "%s\n%s", e.what(), cli.help().c_str());
    return 1;
  }
  if (args.count("help"))
  {
    std::printf("%s", cli.help().c_str());
    return 0;
  }
  Options options;
  options.output = args.count("output") ? args["output"].as<std::string>() : default_output();
  if (const char * cwd = std::getenv("BUILD_WORKING_DIRECTORY"); cwd && std::filesystem::path(options.output).is_relative())
    options.output = (std::filesystem::path(cwd) / options.output).string();
  options.use_sim_time = args["use-sim-time"].as<bool>();
  options.duration = args["duration"].as<double>();
  options.raw = args["raw"].as<bool>();
  options.jpeg_quality = args["jpeg-quality"].as<int>();
  options.depth_format = args["depth-format"].as<std::string>();
  options.png_level = args["png-level"].as<int>();
  options.encode_threads = std::max(1, args["encode-threads"].as<int>());
  options.buffer_bytes = args["buffer-mb"].as<size_t>() << 20;
  options.queue_depth = std::max(1, args["queue-depth"].as<int>());
  options.topics = args["topics"].as<std::vector<std::string>>();
  options.stats_period = args["stats-period"].as<double>();
  options.max_file_bytes = args["max-file-mb"].as<uint64_t>() << 20;
  options.episodic = args["episodic"].as<bool>();
  options.voice = !args["no-voice"].as<bool>();
  options.test_stall = args["test-write-stall"].as<double>();
  options.test_stall_period = args["test-write-stall-period"].as<double>();
  if (options.depth_format != "png" && options.depth_format != "rvl")
  {
    std::fprintf(stderr, "--depth-format must be png or rvl\n");
    return 1;
  }
  if (std::filesystem::exists(options.output))
  {
    std::fprintf(stderr, "%s exists; pick another --output\n", options.output.c_str());
    return 1;
  }
  std::filesystem::create_directories(std::filesystem::path(options.output).parent_path());

  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::SigInt);
  auto recorder = std::make_shared<Recorder>(options);
  // Two threads: the subscriptions and timers (one callback group), and the episode services (their own), so
  // finalizing a bag never holds up receiving.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(recorder);
  std::thread spinner([&executor] { executor.spin(); });
  while (rclcpp::ok() && !recorder->expired()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  executor.cancel();
  spinner.join();
  executor.remove_node(recorder);
  const uint64_t lost = recorder->finish();
  const bool empty = recorder->num_topics() == 0;
  recorder.reset();
  rclcpp::shutdown();
  if (empty)
  {
    std::fprintf(stderr, "Nothing was recorded: no controller state, image or --topics topic was published.\n");
    return 3;
  }
  return lost ? 2 : 0;
}
