// Real-hardware check of rebot_robstride_protocol: drives RobStrideProtocol (joint1-6) and
// RebotGripperProtocol (the gripper, joint7, in m) over SocketCAN the way
// shared_can_ros2's CanSystem does on activation (read initial states, readiness check, enable),
// prints the joint states and disables the motors. It sends no position commands.
//
// Usage: hardware_test [can_interface] [--no-gripper]   (interface default: $CAN_INTERFACE, else can0)
// WARNING: enables the motors; they hold their current position until disabled.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "rebot_robstride_protocol/rebot_gripper_protocol.hpp"
#include "rebot_robstride_protocol/robstride_codec.hpp"
#include "rebot_robstride_protocol/robstride_protocol.hpp"
#include "shared_can_interface/protocol_interface.hpp"
#include "shared_can_ros2/enable_readiness.hpp"
#include "shared_can_ros2/socketcan_transport.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using shared_can_interface::v1::CanFrame;
using shared_can_interface::v1::DeviceConfiguration;
using shared_can_interface::v1::ProtocolInterface;

struct Joint {
  const char *name;
  const char *motor_id;
  const char *model;
  const char *kp;
  const char *kd;
};

// The reBot Arm B601-RS joints (rebot_arm.real_description / rebot_arm_rs.ros2_control.xacro).
const Joint kJoints[] = {
    {"joint1", "0x01", "rs-06", "50.0", "3.0"},
    {"joint2", "0x02", "rs-06", "150.0", "10.0"},
    {"joint3", "0x03", "rs-06", "150.0", "10.0"},
    {"joint4", "0x04", "rs-00", "50.0", "5.0"},
    {"joint5", "0x05", "rs-00", "50.0", "4.0"},
    {"joint6", "0x06", "rs-00", "50.0", "4.0"},
};
const std::vector<std::string> kStates = {"position", "velocity", "effort"};
const std::vector<std::string> kGripperStates = {"position", "velocity", "effort", "temperature"};
constexpr const char *kGripper = "joint7";

void send_all(shared_can_ros2::SocketCanTransport &transport,
              const std::vector<CanFrame> &frames) {
  for (const auto &frame : frames) {
    transport.send(frame);
  }
}

// Dispatches received frames to every protocol (each ignores the others' motors) for `duration`.
void receive_for(shared_can_ros2::SocketCanTransport &transport,
                 const std::vector<ProtocolInterface *> &protocols,
                 std::chrono::milliseconds duration) {
  const auto until = Clock::now() + duration;
  while (Clock::now() < until) {
    CanFrame frame;
    if (transport.receive(frame, std::chrono::milliseconds(20))) {
      for (auto *protocol : protocols) {
        protocol->process_frame(frame, Clock::now());
      }
    }
  }
}

void print_arm(const char *title, const ProtocolInterface &protocol) {
  std::printf("%s\n  %-7s %10s %10s %10s %s\n", title, "joint", "pos[rad]",
              "vel[rad/s]", "effort[Nm]", "fault");
  for (const auto &joint : kJoints) {
    double values[3] = {0, 0, 0};
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
      ok = protocol.get_state(joint.name, kStates[i], values[i]) && ok;
    }
    if (ok) {
      std::printf("  %-7s %10.4f %10.4f %10.4f %s\n", joint.name, values[0],
                  values[1], values[2], protocol.faulted(joint.name) ? "YES" : "no");
    } else {
      std::printf("  %-7s %10s\n", joint.name, "(no state)");
    }
  }
}

void print_gripper(const ProtocolInterface &protocol) {
  double values[4] = {0, 0, 0, 0};
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    ok = protocol.get_state(kGripper, kGripperStates[i], values[i]) && ok;
  }
  if (ok) {
    std::printf("  %-7s position %.4f m (0 closed .. 0.05 open), velocity %.4f m/s, effort %.2f N, %.1f C, fault %s\n",
                kGripper, values[0], values[1], values[2], values[3],
                protocol.faulted(kGripper) ? "YES" : "no");
  } else {
    std::printf("  %-7s (no state)\n", kGripper);
  }
  std::fflush(stdout);
}

// Reads a few of the motor's own settings (read-only; no frame changes anything): the zero reference mode
// (zero_sta: 0 reports angles in 0..2pi, 1 in -pi..pi), the zero offset and the multi-turn angle. Not every firmware
// answers every id, so a missing reply is just reported.
void print_motor_settings(shared_can_ros2::SocketCanTransport &transport, const char *name,
                          std::uint8_t motor_id) {
  using rebot_robstride_protocol::codec::decode_parameter_response;
  using rebot_robstride_protocol::codec::make_read_parameter;
  struct Item {
    const char *label;
    std::uint16_t id;
    bool is_float;
  };
  const Item items[] = {{"run_mode", 0x7005, false},
                        {"mech_pos", 0x7019, true},
                        {"zero_sta", 0x7029, false},
                        {"add_offset", 0x702B, true}};
  std::printf("  %s settings:", name);
  for (const auto &item : items) {
    transport.send(make_read_parameter(motor_id, 0xFD, item.id));
    bool got = false;
    const auto until = Clock::now() + std::chrono::milliseconds(100);
    while (!got && Clock::now() < until) {
      CanFrame frame;
      if (!transport.receive(frame, std::chrono::milliseconds(10))) {
        continue;
      }
      if (const auto response = decode_parameter_response(frame)) {
        if (response->device_id == motor_id && response->parameter_id == item.id) {
          got = true;
          if (item.is_float) {
            std::printf(" %s=%.4f", item.label,
                        rebot_robstride_protocol::codec::parameter_as_f32(*response));
          } else {
            std::printf(" %s=%u", item.label, response->raw_value[0]);
          }
        }
      }
    }
    if (!got) {
      std::printf(" %s=(no reply)", item.label);
    }
  }
  std::printf("\n");
  std::fflush(stdout);
}

void print_raw_gripper(const ProtocolInterface &protocol, const char *when) {
  double position = 0.0, velocity = 0.0, effort = 0.0;
  if (protocol.get_state("joint7_raw", "position", position) &&
      protocol.get_state("joint7_raw", "velocity", velocity) &&
      protocol.get_state("joint7_raw", "effort", effort)) {
    std::printf("  %-7s raw motor angle %.4f rad %s (0 closed .. 4.73 open), %.3f rad/s, %.2f Nm\n",
                kGripper, position, when, velocity, effort);
  } else {
    std::printf("  %-7s raw motor angle %s: (no state)\n", kGripper, when);
  }
  std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
  std::string can_interface = "can0";
  if (const char *env = std::getenv("CAN_INTERFACE")) {
    can_interface = env;
  }
  bool with_gripper = true;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-gripper") {
      with_gripper = false;
    } else {
      can_interface = argv[i];
    }
  }
  std::printf("CAN interface: %s%s\n", can_interface.c_str(),
              with_gripper ? " (with the gripper)" : "");

  shared_can_ros2::SocketCanTransport transport;
  rebot_robstride_protocol::RobStrideProtocol arm;
  rebot_robstride_protocol::RebotGripperProtocol gripper;
  // The gripper motor's raw angle (rad), read as a plain RobStride joint that is never enabled or commanded:
  // the gripper protocol only reports it as a position in m clamped to the stroke, which hides how far past
  // closed/open the motor really is.
  rebot_robstride_protocol::RobStrideProtocol raw_gripper;
  std::vector<ProtocolInterface *> protocols = {&arm};
  if (with_gripper) {
    protocols.push_back(&gripper);
  }
  std::vector<ProtocolInterface *> receivers = protocols;  // everything that listens (not all are enabled)
  if (with_gripper) {
    receivers.push_back(&raw_gripper);
  }
  bool enabled = false;
  int result = 0;
  try {
    transport.configure({{"can_interface", can_interface}});
    transport.open();

    std::vector<DeviceConfiguration> devices;
    for (const auto &joint : kJoints) {
      devices.push_back(
          {joint.name,
           {{"motor_id", joint.motor_id},
            {"model", joint.model},
            {"direction", "1.0"},
            {"offset", "0.0"},
            {"mit_kp", joint.kp},
            {"mit_kd", joint.kd}},
           {"position", "velocity", "kp", "kd", "t_ff"},
           kStates});
    }
    arm.configure({{"host_id", "0xFD"}}, devices);
    if (with_gripper) {
      // The official RS00 gripper (rebot_arm.real_description): motor 7, SI units over a 0.05 m stroke.
      gripper.configure(
          {{"host_id", "0xFD"}},
          {{kGripper,
            {{"motor_id", "0x07"},
             {"model", "rs-00"},
             {"direction", "1.0"},
             {"offset", "0.0"},
             {"mit_kp", "0.5"},  // rebot_arm.real_description.GRIPPER_KP / GRIPPER_KD
             {"mit_kd", "0.25"},
             {"open_position", "4.73"},
             {"closed_position", "0.0"},
             {"stroke", "0.05"},
             {"default_velocity", "0.0125"}},
            {"position", "velocity"},
            kGripperStates}});
    }
    if (with_gripper) {
      raw_gripper.configure({{"host_id", "0xFD"}},
                            {{"joint7_raw",
                              {{"motor_id", "0x07"},
                               {"model", "rs-00"},
                               {"direction", "1.0"},
                               {"offset", "0.0"},
                               {"mit_kp", "0.0"},
                               {"mit_kd", "0.0"}},
                              {"position"},
                              kStates}});
    }
    std::printf("Configured %zu arm devices\n", devices.size());

    // Initial states.
    for (auto *protocol : receivers) {
      protocol->reset_state();
      send_all(transport, protocol->initial_state_requests());
    }
    const auto all_ready = [&] {
      for (auto *protocol : receivers) {
        if (!protocol->initial_state_ready()) {
          return false;
        }
      }
      return true;
    };
    // A motor now and then drops a read: ask again (every 100 ms) for what is missing, as CanSystem does.
    const auto deadline = Clock::now() + std::chrono::milliseconds(1200);
    auto next_retry = Clock::now() + std::chrono::milliseconds(100);
    int rounds = 1;
    while (Clock::now() < deadline && !all_ready()) {
      receive_for(transport, receivers, std::chrono::milliseconds(20));
      if (Clock::now() >= next_retry && !all_ready()) {
        for (auto *protocol : receivers) {
          if (!protocol->initial_state_ready()) {
            send_all(transport, protocol->initial_state_requests());
          }
        }
        ++rounds;
        next_retry = Clock::now() + std::chrono::milliseconds(100);
      }
    }
    std::printf("Initial state read in %d round(s) of requests\n", rounds);
    for (auto *protocol : receivers) {
      for (const auto &name : protocol->missing_initial_states()) {
        std::printf("ERROR: no initial state from %s\n", name.c_str());
        result = 1;
      }
    }
    print_arm("Initial state (motors not enabled):", arm);
    if (with_gripper) {
      print_gripper(gripper);
      print_raw_gripper(raw_gripper, "before enabling");
      print_motor_settings(transport, "joint6 (motor 6)", 0x06);
      print_motor_settings(transport, "joint7 (gripper, motor 7)", 0x07);
    }

    if (result == 0) {
      // Activate: the same sequence as CanSystem::on_activate.
      shared_can_ros2::wait_for_enable_readiness(protocols, transport);
      for (auto *protocol : protocols) {
        for (const auto &batch : protocol->enable_sequence()) {
          enabled = true;
          send_all(transport, batch.frames);
          if (batch.delay_after.count() > 0) {
            std::this_thread::sleep_for(batch.delay_after);
          }
        }
      }
      std::printf("Motors enabled\n");

      // Read: poll the motors for a second.
      for (int i = 0; i < 50; ++i) {
        for (auto *protocol : protocols) {
          if (const auto request = protocol->periodic_request()) {
            transport.send(*request);
          }
        }
        receive_for(transport, receivers, std::chrono::milliseconds(20));
      }
      print_arm("State after activation:", arm);
      if (with_gripper) {
        print_gripper(gripper);
        print_raw_gripper(raw_gripper, "after enabling");
      }
      for (const auto &joint : kJoints) {
        if (arm.faulted(joint.name)) {
          std::printf("ERROR: %s reports a fault\n", joint.name);
          result = 1;
        }
      }
      if (with_gripper && gripper.faulted(kGripper)) {
        std::printf("ERROR: %s reports a fault\n", kGripper);
        result = 1;
      }
    }
  } catch (const std::exception &error) {
    std::printf("ERROR: %s\n", error.what());
    result = 1;
  }

  if (enabled) {
    try {
      for (auto *protocol : protocols) {
        send_all(transport, protocol->disable_frames());
      }
      std::printf("Motors disabled\n");
    } catch (const std::exception &error) {
      std::printf("ERROR disabling motors: %s\n", error.what());
      result = 1;
    }
  }
  std::printf(result == 0 ? "PASSED\n" : "FAILED\n");
  return result;
}
