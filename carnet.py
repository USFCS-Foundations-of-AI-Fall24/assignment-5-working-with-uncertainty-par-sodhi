from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],[0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent": ["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


def main():
    # Associating the parameters with the model structure
    car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_key_present, cpd_battery, cpd_moves)

    car_infer = VariableElimination(car_model)

    print("Query: What is the probability that the car moves given the radio turns on and the car starts?")
    print(car_infer.query(variables=["Moves"], evidence={"Radio": "turns on", "Starts": "yes"}))

    #Given that the car will not move, what is the probability that the battery is not working?
    print("Query: Given that the car will not move, what is the probability that the battery is not working?")
    print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))

    #Given that the radio is not working, what is the probability that the car will not start?
    print("Query: Given that the radio is not working, what is the probability that the car will not start?")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))

    #Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?
    print("Query: Given that the battery is working, does the probability of the radio working change if we discover that the car has gas in it?")
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}))

    #Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?
    print("Query: Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}))

    #What is the probability that the car starts if the radio works and it has gas in it?
    print("Query: What is the probability that the car starts if the radio works and it has gas in it?")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))

    #Probability that the key is not present given that the car does not move
    print("Query: What is the probability that the key is not present given that the car does not move?")
    print(car_infer.query(variables=["KeyPresent"], evidence={"Moves": "no"}))


if __name__ == "__main__":
    main()

