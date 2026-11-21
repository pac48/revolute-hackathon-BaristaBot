// Sets the reBot gripper motor's (motor 7) zero once, at its closed stop, so the driver's "0 = closed" holds across
// power cycles and runs.
//
// Usage: gripper_zero [can_interface] [--write] [--motor N]      (interface default: $CAN_INTERFACE, else can0)
//
// Without --write it only reads: it watches the motor's multi-turn angle for 20 s (move the gripper by hand, motors
// are not enabled) and prints its settings and the angle range seen, so you can see where closed and open are.
//
// With --write (CLOSE THE GRIPPER BY HAND TO ITS STOP FIRST) it
//   1. disables the motor,
//   2. writes zero_sta = 0 (0x7029: the angle is reported in 0..2pi, not -pi..pi; the gripper's ~5 rad of travel
//      does not fit in +-pi, so there the reading folded depending on where the gripper was at power-up),
//   3. sets the mechanical zero here (communication type 6),
//   4. saves the parameters to the motor's flash (type 22), and
//   5. reads the angle and the settings back.
// The motor never moves: it is disabled throughout.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

#include "rebot_robstride_protocol/robstride_codec.hpp"
#include "shared_can_interface/protocol_interface.hpp"
#include "shared_can_ros2/socketcan_transport.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using shared_can_interface::v1::CanFrame;
namespace codec = rebot_robstride_protocol::codec;

constexpr std::uint8_t kHost = 0xFD;
constexpr std::uint16_t kRunMode = 0x7005, kMechPos = 0x7019, kZeroSta = 0x7029, kAddOffset = 0x702B;

// Reads one parameter (the 4 raw bytes), or false if the motor doesn't answer.
bool read_parameter(shared_can_ros2::SocketCanTransport &transport, std::uint8_t motor,
                    std::uint16_t id, codec::ParameterResponse &out) {
  transport.send(codec::make_read_parameter(motor, kHost, id));
  const auto until = Clock::now() + std::chrono::milliseconds(150);
  while (Clock::now() < until) {
    CanFrame frame;
    if (!transport.receive(frame, std::chrono::milliseconds(10))) {
      continue;
    }
    if (const auto response = codec::decode_parameter_response(frame)) {
      if (response->device_id == motor && response->parameter_id == id) {
        out = *response;
        return true;
      }
    }
  }
  return false;
}

bool read_angle(shared_can_ros2::SocketCanTransport &transport, std::uint8_t motor, double &angle) {
  codec::ParameterResponse response;
  if (!read_parameter(transport, motor, kMechPos, response)) {
    return false;
  }
  angle = codec::parameter_as_f32(response);
  return true;
}

void print_settings(shared_can_ros2::SocketCanTransport &transport, std::uint8_t motor, const char *when) {
  codec::ParameterResponse run, zero, offset;
  const bool r = read_parameter(transport, motor, kRunMode, run);
  const bool z = read_parameter(transport, motor, kZeroSta, zero);
  const bool o = read_parameter(transport, motor, kAddOffset, offset);
  double angle = 0.0;
  const bool a = read_angle(transport, motor, angle);
  std::printf("%s: run_mode=%s zero_sta=%s add_offset=%s angle=%s\n", when,
              r ? std::to_string(run.raw_value[0]).c_str() : "(no reply)",
              z ? std::to_string(zero.raw_value[0]).c_str() : "(no reply)",
              o ? std::to_string(codec::parameter_as_f32(offset)).c_str() : "(no reply)",
              a ? std::to_string(angle).c_str() : "(no reply)");
  std::fflush(stdout);
}

CanFrame frame_of(std::uint8_t type, std::uint8_t motor) {
  CanFrame frame;
  frame.extended = true;
  frame.arbitration_id =
      codec::build_extended_id(static_cast<codec::CommunicationType>(type), kHost, motor);
  frame.dlc = 8;
  frame.data.fill(0);
  return frame;
}

} // namespace

int main(int argc, char **argv) {
  std::string can_interface = "can0";
  if (const char *env = std::getenv("CAN_INTERFACE")) {
    can_interface = env;
  }
  bool write = false;
  std::uint8_t motor = 7;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--write") {
      write = true;
    } else if (arg == "--motor" && i + 1 < argc) {
      motor = static_cast<std::uint8_t>(std::stoi(argv[++i]));
    } else {
      can_interface = arg;
    }
  }
  std::printf("CAN interface %s, motor %u, %s\n", can_interface.c_str(), motor,
              write ? "WRITING the zero" : "read-only");

  try {
    shared_can_ros2::SocketCanTransport transport;
    transport.configure({{"can_interface", can_interface}});
    transport.open();

    print_settings(transport, motor, "now");
    if (!write) {
      std::printf("Watching the angle for 20 s: move the gripper by hand to closed and to open.\n");
      double lo = 1e9, hi = -1e9, last = 0.0;
      const auto until = Clock::now() + std::chrono::seconds(20);
      while (Clock::now() < until) {
        if (read_angle(transport, motor, last)) {
          lo = std::min(lo, last);
          hi = std::max(hi, last);
          std::printf("\r  angle %8.4f rad   seen %8.4f .. %8.4f ", last, lo, hi);
          std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      std::printf("\nDry run: nothing was written. Close the gripper to its stop and run again with --write.\n");
      return 0;
    }

    // Disable first: the zero can only be set on a stopped motor.
    transport.send(codec::make_disable(motor, kHost, false));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    codec::ParameterResponse run;
    if (!read_parameter(transport, motor, kRunMode, run)) {
      std::printf("ERROR: motor %u does not answer\n", motor);
      return 1;
    }

    transport.send(codec::make_write_parameter_i8(motor, kHost, kZeroSta, 0));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto set_zero = frame_of(static_cast<std::uint8_t>(codec::CommunicationType::kSetZeroPosition), motor);
    set_zero.data[0] = 1;
    transport.send(set_zero);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto save = frame_of(static_cast<std::uint8_t>(codec::CommunicationType::kSaveParameters), motor);
    for (std::uint8_t i = 0; i < 8; ++i) {
      save.data[i] = static_cast<std::uint8_t>(i + 1);
    }
    transport.send(save);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    print_settings(transport, motor, "after");
    std::printf("Done. Power-cycle the motor and run without --write to check that the zero stuck.\n");
  } catch (const std::exception &error) {
    std::printf("ERROR: %s\n", error.what());
    return 1;
  }
  return 0;
}
