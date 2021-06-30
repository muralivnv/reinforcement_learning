# Robot Tasker

## Objective

Task of the robot is to go through each some target locations in scene and perform tasks that are available.

## Definition of Task

A task can be anything from vacuuming or picking things off the floor or helping people.  

## Guidelines

Tasks can spawn at any moment in time within a given world bounds. Each task is assigned a priority with 1 being the highest priority and 3 being the lowest priority. Highest priority tasks cannot be delayed for more than $T^{high}_{max}$ mins while the lowest priority tasks can be delayed for a maximum time of $T^{low}_{max}$.  

## System

Robot can travel at a maximum forward speed of $V_{max}$ with an mpc (mileage per charge) of $C$ mpc. Rate of decrease of charge of the robot goes up when robot increases it's speed.

### To-Do

* Learning to drive -- **WIP**
  > Let robot learn how to get from point A to point B in minimum time.

* Learning to prioritize task/charge
  > Let robot learn how to prioritize which tasks to pick and decide whether to go to a charging station or proceed to next task.
  > It would be more optimal if a robot learns to pick a task that is on the way to the charging station increasing it's reward.

* Learning to avoid obstacles
  > Let robot learn how to avoid both moving and stationary obstacles in scene.
