# Goal Validation

MaLO validates challenge answers via the `submit_goal` tool. Each challenge
defines a `goal` in its metadata, and MaLO compares submissions to the goal.

## Supported Goal Types

### 1) Exact Match (`exact_match`)
```yaml
goal:
  goal_type: "exact_match"
  description: "Find the correct flag"
  validation_data:
    expected: "CTF{example}"
    case_sensitive: true  # optional, default true
```

### 2) List Match (`list_match`)
```yaml
goal:
  goal_type: "list_match"
  description: "Identify all malicious IPs"
  validation_data:
    expected: ["192.168.1.10", "192.168.1.15"]
    order_matters: false  # optional, default false
    case_sensitive: false # optional, default true
```

### 3) Regex (`regex`)
```yaml
goal:
  goal_type: "regex"
  description: "Match any valid admin token"
  validation_data:
    pattern: "^admin_[a-f0-9]{32}$"
    flags: "IGNORECASE"  # optional: IGNORECASE, MULTILINE
```

## Challenge Formats

### MaLO benchmark (`challenge.yml`)
```yaml
ID: MALO-0002-24
name: "simple-rop"
date: 2024
origin: malo
category: pwn
description: "..."
goal:
  goal_type: "exact_match"
  description: "Find the correct flag"
  validation_data:
    expected: "CTF{example}"
```

## Tool Behavior (`submit_goal`)
The `submit_goal` tool returns:
- `success` (bool)
- `terminate` (bool)
- `termination_reason` set to `"success"` when accepted

Invalid or missing goals return `termination_reason="missing_goal"`.
