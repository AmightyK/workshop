# Goal Specification - BFMC Warehouse Scenario with V-JEPA Integration

## Role

You are a senior Robotics Engineer specializing in:

* Autonomous Navigation
* Dynamic Obstacle Avoidance
* World Models (V-JEPA / V-JEPA2)
* Behavior Planning
* Model Predictive Decision Making
* ROS2-based Autonomous Systems

Your task is to modify and improve the existing BFMC autonomous vehicle stack.

---

## Current System

The current vehicle already supports:

* Lane following
* Sign detection
* Object detection
* Stable navigation
* Latent extraction using V-JEPA
* Basic stop-and-go obstacle handling

The current performance must be preserved.

Do not degrade:

* Tracking stability
* Lane keeping
* Existing perception performance
* Current inference speed

---

# Mission Scenario

The vehicle starts from the warehouse start point.

Target mission:

1. Navigate to Shelf A.
2. Detect a BLUE package.
3. Pick up the BLUE package.
4. Return to the drop-off zone.
5. Place the package successfully.

---

# Dynamic Human Interaction

## Human #1 (Static Blocking)

Scenario:

* Human stands still.
* Human only begins moving when the vehicle gets close.

Required behavior:

* Detect human.
* Predict occupancy of path.
* Wait safely.
* Resume motion after path becomes free.

Success criteria:

* No collision.
* No aggressive steering.
* No unnecessary rerouting.

---

## Human #2 (Dynamic Crossing)

Scenario:

* Human continuously walks left-right across the lane.

Required behavior:

Instead of only stopping:

1. Track trajectory history.
2. Estimate velocity.
3. Predict future occupancy.
4. Evaluate if safe crossing exists.
5. Decide:

   Decision A:

   * Stop and wait

   Decision B:

   * Overtake / pass safely

Decision must be based on:

* Predicted collision probability
* Time-to-collision
* Predicted free-space window

The planner must explicitly output:

* WAIT
* PASS
* REPLAN

and the reason for the decision.

---

# V-JEPA Integration Requirements

Use V-JEPA not only as a feature extractor.

The system must demonstrate:

## Environment Understanding

Extract latent embeddings during the mission.

Store:

* Raw frame
* Latent vector
* Timestamp
* Vehicle pose

for:

* Normal driving
* Human #1 encounter
* Human #2 encounter
* Shelf approach
* Pick-up operation
* Return path

---

## Predictive Capability Demonstration

Implement latent prediction experiments.

For every critical scene:

Input:

Current latent z_t

Predict:

z_t+1
z_t+2
z_t+3

Compare:

Predicted latent
vs
Actual latent

Metrics:

* L1 latent error
* Cosine similarity
* Prediction drift

Generate plots and logs.

The goal is to demonstrate that V-JEPA captures future scene evolution and environmental dynamics. V-JEPA2 specifically supports latent-space prediction and planning through future latent rollout.

---

## Behavior-Level Prediction Demo

Create examples showing:

Case 1:
Human leaves path

Expected prediction:
Path becomes free

Case 2:
Human continues crossing

Expected prediction:
Path remains occupied

Case 3:
Vehicle can safely pass

Expected prediction:
Collision-free future occupancy

Generate visualizations showing:

* Current frame
* Future latent prediction
* Planner decision

---

# Pick-and-Place Stage

At Shelf A:

1. Detect BLUE package.
2. Align vehicle.
3. Raise lifting mechanism.
4. Grasp package.
5. Verify successful grasp.

Verification:

* Detection confidence
* Position consistency
* Gripper state

If grasp fails:

* Retry alignment
* Retry grasp

Maximum retries configurable.

---

# Navigation Accuracy Improvement

Observed issue:

Second 90-degree turn near Shelf A overshoots.

Required investigation:

1. Path tracking error.
2. Steering latency.
3. Curvature planning.
4. PID tuning.
5. Pure Pursuit lookahead.
6. Stanley controller parameters.
7. Corner apex selection.

Provide:

* Root-cause analysis
* Proposed fix
* Expected improvement
* Validation procedure

Priority:

Eliminate overshoot while preserving overall stability.

---

# Deliverables

Provide:

1. Architecture diagram
2. ROS2 node modifications
3. State machine updates
4. Planner changes
5. V-JEPA integration design
6. Latent logging pipeline
7. Evaluation metrics
8. Test scenarios
9. Failure cases
10. Implementation roadmap

---

# Acceptance Criteria

Mission is successful only if:

* Vehicle reaches Shelf A.
* BLUE package is correctly picked.
* Vehicle returns to drop zone.
* No collision with either human.
* Dynamic decision making is demonstrated.
* V-JEPA latent prediction is logged and evaluated.
* Tracking performance is not worse than current baseline.
* Second 90° corner overshoot is significantly reduced.
