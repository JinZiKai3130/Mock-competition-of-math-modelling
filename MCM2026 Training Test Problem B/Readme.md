# Elevator Operation Data Documentation

This document describes the elevator operation dataset exported from the Group Control System (GCS). 
The dataset consists of 6 CSV files recording information about elevator calls, operation, stops, door status, and load changes.

## 1. Dataset Overview

**File List:**
1. `hall_calls.csv`: Records hall call signals (people pressing up/down buttons in the elevator lobby).
2. `car_calls.csv`: Records car call signals (people pressing floor buttons inside the elevator car).
3. `car_stops.csv`: Records the time when an elevator stops at a floor and the reason for stopping.
4. `car_departures.csv`: Records the time when an elevator departs from a floor.
5. `load_changes.csv`: Records load change data when passengers enter or exit the elevator.
6. `maintenance_mode.csv`: Records when elevators enter or exit maintenance mode.

---

## 2. Common Field Descriptions

All files contain the following basic fields：
*   **Time**：The specific timestamp when the event occurred, precise to seconds, formatted as `YYYY-MM-DD HH:MM:SS`。(Note that Excel does not directly display seconds when opened. Seconds are visible only when clicking on a cell, using text editing, or opening the data with a program.)
*   **Elevator ID**: The unique identifier for the elevator (e.g., `A`、`B`、`C`、`D`)。
*   **Floor**：The floor number where the event occurred, starting from `1`。

---

## 3. Detailed File Specifications

### 3.1 Hall Calls (hall_calls.csv)
Records registration of passenger hall call requests from elevator lobbies.
| Field Name | Description | Example Value |
| :--- | :--- | :--- |
| **Time** | Time when the hall call signal occurred | `2025-11-01 08:30:15` |
| **Elevator ID** | Elevator number that responded to the call | `A` |
| **Direction** | Desired travel direction (`Up` or `Down`) | `Up` |
| **Floor** | Floor where the passenger is located | `6` |

> **Scenario Example**:
>
> | Time | Elevator ID | Direction | Floor | Explanation  |
> | ---- | ---- | ---- | ---- |---------------------- |
> | 2025/11/01 00:33:04 | C    | Down | 10   | Down hall call task assigned to elevator C for floor 10 |
> | 2025/11/01 00:33:22 | C    | Down | 3,10 | Floor 3 added to elevator C's down hall call task |
> | 2025/11/01 00:33:30 | C    | Down | 3    | Floor 10 completed (or possibly canceled/transferred) for elevator C's down hall call task |
> | 2025/11/01 00:33:55 | C    | Down |      | Floor 3 completed (or possibly canceled/transferred) for elevator C's down hall call task |

### 3.2 Car Calls (car_calls.csv)
Records registration of passenger button presses inside the elevator car.

| Field Name | Description | Example Value |
| :------- | :------------- | :----------- |
| **Time** | Time when the button action occurred | `2025-11-01 08:31:05` |
| **Elevator ID** | Elevator number where the call was made | `A` |
| **Floor** | Destination floor pressed by the passenger | `15` |
| **Action** | Button status change (`Register` indicates press activation, `Cancel` indicates signal clearance) | `Register` |

> **Scenario Example**:
>
> | Time | Elevator ID | Floor | Action | Explanation  |
> | ------------------- | ---- | ---- | ---- | -------------------------------------- |
> | 2025/11/01 06:59:16 | A    | 6    | Register | Floor 6 car call registered for elevator A   |
> | 2025/11/01 06:59:16 | A    | 7    | Register | Floor 7 car call registered for elevator A  |
> | 2025/11/01 06:59:53 | A    | 6    | Cancel | Floor 6 car call completed (or possibly canceled) for elevator A |
> | 2025/11/01 07:00:11 | A    | 7    | Cancel | Floor 7 car call completed (or possibly canceled) for elevator A |

### 3.3 Car Stops (car_stops.csv)

Records the time when an elevator stops at a floor and the reason for stopping.

| Field Name  | Description   | Example Value |
| :------------------- | :-------------------------------------- | :-------------------- |
| **Time**   | Time when the elevator completely stopped | `2025-11-01 08:30:25` |
| **Elevator ID** | Elevator number   | `A`  |
| **Floor**             | Floor where the elevator stopped  | `6`  |
| **Direction**   | Direction status when the elevator stopped   | `Up` |
| **Stop Reason - Car Call** | Whether stopped because someone inside pressed this floor (`Yes`/`No`) | `No`   |
| **Stop Reason - Hall Call** | Whether stopped because someone on this floor called the elevator (`Yes`/`No`)  | `Yes`   |
| **Stop Reason - Idle**     | Whether stopped because idle with no tasks (`Yes`/`No`)   | `No`  |

> **Scenario Example**:
>
> | Time                | Elevator ID | Floor | Direction | Stop Reason - Car Call | Stop Reason - Hall Call | Stop Reason - Idle | Explanation                           |
> | ------------------- | ---- | ---- | ---- | ---------------- | ---------------- | ------------ | ---------------------------------- |
> | 2025/11/01 00:00:25 | A    | 6    | Down | No               | Yes               | No           | Elevator A going down, stopped at floor 6 due to hall call |

### 3.4 Car Departures (car_departures.csv)

Records the time when an elevator completes stopping, closes doors, and begins moving to the next destination.

| Field Name | Description | Example Value |
| :--- | :--- | :--- |
| **Time** | Time when the elevator starts departing | `2025-11-01 08:30:45` |
| **Elevator ID** | Elevator number | `A` |
| **Floor** | Floor from which the elevator departs | `6` |

> **Scenario Example**:
>
> | Time                | Elevator ID | Floor | Explanation                     |
> | ------------------- | ---- | ---- | ---------------------------- |
> | 2025/11/01 00:00:16 | A    | 3    | Elevator A departs from floor 3, heading to next floor |

### 3.5 Load Changes (load_changes.csv)
Records load changes during elevator stops, used to estimate passenger count entering/exiting.
*Note: Load has been converted from percentage to kilograms based on rated load (2100kg).*

| Field Name | Description | Example Value |
| :--- | :--- | :--- |
| **Time** | Time when load was detected | `2025-11-01 08:30:35` |
| **Elevator ID** | Elevator number | `A` |
| **Floor** | Floor where load change occurred | `6` |
| **Load In (kg)** | Weight increase from passengers entering (in `kg`), may have errors | `140.5` |
| **Load Out (kg)** | Weight decrease from passengers exiting (in `kg`), may have errors | `0.0` |

> **Scenario Example**:
>
> | Time                | Elevator ID | Floor | Load In (kg) | Load Out (kg) | Explanation                                      |
> | ------------------- | ---- | ---- | ---------------- | ---------------- | --------------------------------------------- |
> | 2025/11/01 05:20:41 | A    | 1    | 105              | 0                | 105kg load entered elevator A at floor 1, no load exited |

### 3.6 Maintenance Mode (maintenance_mode.csv)
Records when elevators enter or exit maintenance mode. In this mode, elevators may not respond to normal calls.

| Field Name | Description | Example Value |
| :--- | :--- | :--- |
| **Time** | Time when status changed | `2025-11-01 14:00:42` |
| **Elevator ID** | Elevator number | `D` |
| **Mode** | Fixed as "Maintenance Light" | `Maintenance Light` |
| **Action** | Entering or exiting this mode (`Enter`/`Exit`) | `Enter` |

> **Scenario Example**:
>
> | Time                | Elevator ID | Mode       | Action | Explanation            |
> | ------------------- | ---- | ---------- | ---- | ------------------- |
> | 2025/11/05 14:02:17 | I    | Maintenance Light | Enter | Elevator I enters maintenance mode |
> | 2025/11/05 14:08:41 | I    | Maintenance Light | Exit | Elevator I exits maintenance mode |