## Resistor network environment

The task for the agent would be to hypothesis the network topology and compute resistance of some unknown resistors. It would receive some external measurements (voltage, current, power at different terminals) maybe also partial information about internal components

Hypothesis that the model has to make is about topology and resistor values.

Tools would be series resistance calculator, parallel resistance calculator, some circuit simulator, some validation tool etc.


### Scoring


`check_resistor_topology` - This functions check if the topolgy proposed is correct to that of the expected one irrespective of the order in which predicted connection is given

`check_resistance_measurements` - This function creates a scoring system to validate whether a circuit topology matches expected resistance measurements.

```python
# Define what we expect to measure
expected_measurements = [
    {"node_a": "A", "node_b": "B", "resistance": 10.0},  # Between A-B: 10Ω
    {"node_a": "B", "node_b": "C", "resistance": 15.0},  # Between B-C: 15Ω
    {"node_a": "A", "node_b": "C", "resistance": 25.0},  # Between A-C: 25Ω
]

# Create the scorer
scorer = check_resistance_measurements(expected_measurements, tolerance=0.05)

# Test a topology (JSON format)
topology = {
    "resistors": {"R1": 10.0, "R2": 15.0},
    "connections": [
        ["A", "B", "R1"],  # 10Ω resistor between A and B
        ["B", "C", "R2"],  # 15Ω resistor between B and C
    ],
}

# Get score (should be close to 1.0 if topology matches expectations)
score = scorer(json.dumps(topology))
```

This function internally uses  `_simulate_resistance` , which performs nodal analysis to find resistance between two points.
