import numpy as np
from common.numpy_fast import interp

import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
import math

# One, two and three bar distances (in s)
ONE_BAR_DISTANCE = 0.8  # in seconds
TWO_BAR_DISTANCE = 1.2  # in seconds
THREE_BAR_DISTANCE = 1.8  # in seconds
FOUR_BAR_DISTANCE = 2.5   # in seconds

# Variables that change braking profiles
CITY_SPEED = 19.44  # braking profile changes when below this speed based on following dynamics below [m/s]
GAP_CLOSURE_SPEED = -1  # relative velocity between you and lead car which activates braking profile change [m/s]
RAPID_GAP_CLOSURE_SPEED = -2.5  # relative velocity between you and lead car which activates a broking profile change + RAPID_DELTA [m/s]
RAPID_DELTA = 0.3  # increased braking profile for approaching lead car at RAPID_GAP_CLOSURE_SPEED [s]
TAILGATE_DISTANCE = 17.5  # when below this distance between you and lead car, braking profile change is active based on PULLAWAY_REL_V [m]
PULLAWAY_REL_V = 0.25  # within TAILGATE_DISTANCE, if the car is pulling away w/ rel velocity that exceeds this value, then change BACK to set bar distance [m/s]
MIN_DISTANCE = 7  # keep a minimum distance between you and lead car (when below this, activates braking profile change) [m]
STOPPING_DISTANCE = 2  # increase distance from lead car when stopped

# Braking profile changes (makes the car brake harder because it wants to be farther from the lead car - increase to brake harder)
BRAKING_ONE_BAR_DISTANCE = 2.3  # more aggressive braking when using one bar distance by increasing follow distance [s]
BRAKING_TWO_BAR_DISTANCE = 2.3  # more aggressive braking when using two bar distance by increasing follow distance [s]
BRAKING_THREE_BAR_DISTANCE = 2.1  # no change in braking profile

class LongitudinalMpc(object):
  def __init__(self, mpc_id, live_longitudinal_mpc):
    self.live_longitudinal_mpc = live_longitudinal_mpc
    self.mpc_id = mpc_id

    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False

    self.override = False
    self.lastTR = 2
    self.last_cloudlog_t = 0.0
    self.v_rel = 10
    self.last_cloudlog_t = 0.0
    self.tailgating = 0
    self.street_speed = 0
    self.lead_car_gap_shrinking = 0
    self.lead_car_rapid_gap_shrinking = 0

  def send_mpc_solution(self, qp_iterations, calculation_time):
    qp_iterations = max(0, qp_iterations)
    dat = messaging.new_message()
    dat.init('liveLongitudinalMpc')
    dat.liveLongitudinalMpc.xEgo = list(self.mpc_solution[0].x_ego)
    dat.liveLongitudinalMpc.vEgo = list(self.mpc_solution[0].v_ego)
    dat.liveLongitudinalMpc.aEgo = list(self.mpc_solution[0].a_ego)
    dat.liveLongitudinalMpc.xLead = list(self.mpc_solution[0].x_l)
    dat.liveLongitudinalMpc.vLead = list(self.mpc_solution[0].v_l)
    dat.liveLongitudinalMpc.cost = self.mpc_solution[0].cost
    dat.liveLongitudinalMpc.aLeadTau = self.a_lead_tau
    dat.liveLongitudinalMpc.qpIterations = qp_iterations
    dat.liveLongitudinalMpc.mpcId = self.mpc_id
    dat.liveLongitudinalMpc.calculationTime = calculation_time
    self.live_longitudinal_mpc.send(dat.to_bytes())

  def setup_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.mpc_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, CS, lead, v_cruise_setpoint):
    v_ego = CS.carState.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = max(0, lead.dRel - STOPPING_DISTANCE)  # increase stopping distance to car by X [m]
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = max(lead.aLeadTau, (a_lead ** 2 * math.pi) / (2 * (v_lead + 0.01) ** 2))
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.libmpc.init_with_simulation(self.v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      v_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU
      x_lead = 50.0

    # Calculate conditions
    self.v_rel = v_lead - v_ego   # calculate relative velocity vs lead car

    # Is the car running surface street speeds?
    if v_ego < CITY_SPEED:
      self.street_speed = 1
    else:
      self.street_speed = 0

    # Is the gap from the lead car shrinking?
    if self.v_rel < GAP_CLOSURE_SPEED:
      self.lead_car_gap_shrinking = 1
    else:
      self.lead_car_gap_shrinking = 0

    # Is the car tailgating the lead car?
    if x_lead < MIN_DISTANCE or (x_lead < TAILGATE_DISTANCE and self.v_rel < PULLAWAY_REL_V):
      self.tailgating = 1
    else:
      self.tailgating = 0

    # Calculate mpc
    # Adjust distance from lead car when distance button pressed
    if CS.carState.readdistancelines == 1:
      #if self.street_speed and (self.lead_car_gap_shrinking or self.tailgating):
      if self.street_speed and (self.lead_car_gap_shrinking or self.tailgating):
        TR = BRAKING_ONE_BAR_DISTANCE
        if self.lead_car_rapid_gap_shrinking:
          TR = TR + RAPID_DELTA  # add more braking if lead car is coming in fast
      else:
        TR = ONE_BAR_DISTANCE
      if CS.carState.readdistancelines != self.lastTR:
        self.libmpc.init(MPC_COST_LONG.TTC, 1.0, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.lastTR = CS.carState.readdistancelines

    elif CS.carState.readdistancelines == 2:
      #if self.street_speed and (self.lead_car_gap_shrinking or self.tailgating):
      if self.street_speed and (self.lead_car_gap_shrinking or self.tailgating):
        TR = BRAKING_TWO_BAR_DISTANCE
        if self.lead_car_rapid_gap_shrinking:
          TR = TR + RAPID_DELTA  # add more braking if lead car is coming in fast
      else:
        TR = TWO_BAR_DISTANCE
      if CS.carState.readdistancelines != self.lastTR:
        self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.lastTR = CS.carState.readdistancelines

    elif CS.carState.readdistancelines == 3:
      # if self.street_speed:
      #if self.street_speed and (self.lead_car_gap_shrinking or self.tailgating):
      if self.street_speed and (self.lead_car_gap_shrinking or self.tailgating):
        TR = BRAKING_THREE_BAR_DISTANCE
        if self.lead_car_rapid_gap_shrinking:
          TR = TR + RAPID_DELTA  # add more braking if lead car is coming in fast
      else:
        TR = THREE_BAR_DISTANCE
      if CS.carState.readdistancelines != self.lastTR:
        self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.lastTR = CS.carState.readdistancelines

    elif CS.carState.readdistancelines == 4:
      TR = FOUR_BAR_DISTANCE
      if CS.carState.readdistancelines != self.lastTR:
        self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.lastTR = CS.carState.readdistancelines

    else:
     TR = TWO_BAR_DISTANCE # if readdistancelines != 1,2,3,4
     self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    t = sec_since_boot()
    n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead, TR)
    duration = int((sec_since_boot() - t) * 1e9)
    self.send_mpc_solution(n_its, duration)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]

    # Reset if NaN or goes through lead car
    dls = np.array(list(self.mpc_solution[0].x_l)) - np.array(list(self.mpc_solution[0].x_ego))
    crashing = min(dls) < -50.0
    nans = np.any(np.isnan(list(self.mpc_solution[0].v_ego)))
    backwards = min(list(self.mpc_solution[0].v_ego)) < -0.01

    if ((backwards or crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.mpc_id, backwards, crashing, nans))

      self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                       MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0
      self.v_mpc = v_ego
      self.a_mpc = CS.carState.aEgo
      self.prev_lead_status = False
