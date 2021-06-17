# Task Robot

## Objective

Task of the robot is to go through each area in scene and perform tasks that are available.

## Definition of Task

A task can be anything from vacuuming/picking things off the floor/helping people.  

## Guidelines

Tasks can spawn at any moment in time and each task is assigned a priority with 1 being the lowest priority
and 3 being the highest priority. Highest priority tasks cannot be delayed for more than 15 mins while the lowest priority tasks can be delayed for a maximum time of 25 mins.  
If a robot is currently in the middle of a task, the robot can choose whether to complete or hold the current task to take care of highest priority task.  

## Robot

Robot can travel at a maximum forward speed of 2 m/sec with an mpc (mileage per charge) of 5 Km. It takes roughly 10 mins to go from 0 to 100% of battery capacity. Multiple charging points are available in the task area.

## Problem Setup

### Reward

* For each second a robot doesn't reach a task space a reward of $r_a$ is given
* Once a robot reaches the task space a reward of $r_t$ is given
* Each percentage drop in battery between tasks will result in a reward of $r_b$

### State-Space

At each time-step robot will be in the state of,

* current position
* current battery percentage
* task state

### Action-Space

At each time-step robot can choose

* selecting a task
* travel to a task location
* travel to a charge station

### To-Do revised

* Learning to drive -- `WIP`
* Learning to avoid obstacles
* Learning to prioritize task/charge

#### To-Do today

* ~~Create a way to sample trajectories based on exploration~~
* Have this exploration setup in such a way as the loss goes down, exploration should go down as well

* ~~Setup target network~~
  * ~~Define parameter to do soft-update of target critic network~~

* ~~Make sure all functions follow input and output mechanism properly~~

* Make code look pretty | good for now

* Cleanup `ANN_activation.h`
