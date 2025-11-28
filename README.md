# Distributed Traffic Sketch Simulation
A lightweight simulation of real-time network traffic analytics using streaming sketch algorithms:
- Morris++ Counter – approximate total packet count
- Flajolet–Martin (FM) – estimate number of unique IPs
- Count–Min Sketch (CMS) – per-IP frequency + heavy hitters
- AMS Sketch (Improved) – estimate second moment F₂
The system generates a synthetic packet stream with heavy hitters and processes it in one pass using sub-linear memory.

---

## Features
- Fast single-pass analytics
- Very small memory footprint
- Improved AMS sketch with median-of-means
- Detailed accuracy report for all estimators

---

## Files
- main.py – entry point
- simulation.py – runs generator + sketches
- morris.py, flajolet_martin.py, count_min.py, ams.py – algorithms
- generator.py – synthetic traffic stream
- utils.py – hashing utilities

---

## Team
| S. No | Name       | Roll No  |
| :---- | :--------- | :------- |
|   1   | Akshath RH | CS22B003 |
|   2   | M Akash    | CS22B037 |
