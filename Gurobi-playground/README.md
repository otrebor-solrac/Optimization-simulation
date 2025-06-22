# Optimization Problems Solved with Gurobi

This repository contains Python code examples demonstrating how to solve various optimization problems using the Gurobi Optimizer. Each example showcases different aspects of optimization modeling, from nonlinear problems requiring specialized handling to linear programs with detailed sensitivity analysis.

## Table of Contents

1.  [Open-Top Box Optimization](#open-top-box-optimization)
2.  [Corner Passage Optimization](#corner-passage-optimization)
3.  [General Linear Programming Analysis](#general-linear-programming-analysis)

---

## Open-Top Box Optimization

This section explores a classic optimization challenge: determining the dimensions of an open-top box with a square base that minimize its material usage (surface area) while maintaining a specific volume. This problem introduces **nonlinearities** in both the objective function and constraints, requiring Gurobi's advanced capabilities for non-convex optimization.

**Key Learnings:**
* Formulating nonlinear objectives and constraints.
* Utilizing Gurobi's `NonConvex` parameter for non-convex quadratic problems.
* Interpreting optimal dimensions for practical design.

---

## Corner Passage Optimization

This example tackles a geometric optimization problem: finding the maximum length of a rigid object that can be maneuvered horizontally around a right-angle hallway corner where the width changes. The solution involves applying **geometric principles** and translating them into a **nonlinear optimization model**.

**Key Learnings:**
* Modeling real-world geometric constraints as mathematical equations.
* Deriving objective functions based on physical properties (length).
* Applying Gurobi to solve problems with embedded geometric relationships.

---

## General Linear Programming Analysis

This part of the repository provides a comprehensive analysis of a generic **linear programming model**. Beyond just finding the optimal solution, it delves into crucial **sensitivity analysis** techniques. Understanding sensitivity is vital for assessing the robustness of an optimal solution and gaining insights into how changes in input parameters (e.g., costs, resource availability) might affect the outcome.

**Key Learnings:**
* Setting up and solving standard linear programming problems.
* Interpreting optimal variable values and the overall objective.
* Understanding and utilizing **reduced costs**: how much objective coefficients can change before a non-basic variable becomes active.
* Analyzing **slack variables**: identifying unused resources or capacities.
* Grasping **dual prices (shadow prices)**: quantifying the value of an additional unit of a constrained resource.
* Exploring **sensitivity ranges for objective coefficients and right-hand sides**: determining the limits within which model parameters can vary without changing the optimal solution's structure.