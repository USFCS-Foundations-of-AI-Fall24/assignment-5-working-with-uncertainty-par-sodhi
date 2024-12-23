import random
import argparse
import codecs
import os
import numpy
import numpy as np


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions


        ## part 1 - you do this.
    def load(self, basename):
        """Reads HMM structure from transition and emission files, as well as the probabilities."""
        with open(f"{basename}.trans", "r") as trans_file:
            for line in trans_file:
                parts = line.strip().split()
                from_state = parts[0]
                if not from_state in self.transitions:
                    self.transitions[from_state] = {}
                i = 1
                while i < len(parts):
                    to_state = parts[i]
                    probability = float(parts[i + 1])
                    self.transitions[from_state][to_state] = probability
                    i += 2

        with open(f"{basename}.emit", "r") as emit_file:
            for line in emit_file:
                parts = line.strip().split()
                state = parts[0]
                if not state in self.emissions:
                    self.emissions[state] = {}
                i = 1
                while i < len(parts):
                    observation = parts[i]
                    probability = float(parts[i + 1])
                    self.emissions[state][observation] = probability
                    i += 2

    ## you do this.
    # Source I used: https://www.geeksforgeeks.org/hidden-markov-model-in-machine-learning/
    def generate(self, n):
        """Return an n-length Sequence by randomly sampling from this HMM."""
        state_path = []
        emission_symbols = []
        state = random.choices(
            population=list(self.transitions["#"].keys()),
            weights=list(self.transitions["#"].values()),
            k=1
        )[0]
        state_path.append(state)

        while len(emission_symbols) < n:
            state = random.choices(
                population=list(self.transitions[state].keys()),
                weights=list(self.transitions[state].values()),
                k=1
            )[0]
            state_path.append(state)
            if state in self.emissions:
                output = random.choices(
                    population=list(self.emissions[state].keys()),
                    weights=list(self.emissions[state].values()),
                    k=1
                )[0]
                emission_symbols.append(output)

        return Sequence(state_path[:n], emission_symbols)

    # Used forward psuedo code to implement this
    #Set up the initial matrix M, with P=1.0 for the ‘#’ state.
    # For each state on day 1: P(state | e0) = ￼P(e0 | state) P(state | #)
    # for i = 2 to n  :
    # foreach state s:
        # sum = 0
        # for s2 in states :
        # sum += M[s2, i-1]*T[s2,s]*E[O[i],s]
    # M[s2,i] = sum
    def forward(self, sequence):
        states = list(self.transitions.keys())
        states.remove("#")
        safe_landings = {"2,5", "3,4", "4,3", "4,4","5,5"}

        # Set up the initial matrix M, with P=1.0 for the ‘#’ state.
        M = np.zeros((len(states), len(sequence)))

        # For each state on day 1: P(state | e0) = P(e0 | state) P(state | #)
        self.initialize_first_observation(sequence, states, M)

        for i in range(1, len(sequence)):
            for state in range(len(states)):
                sum = 0
                for second_state in range(len(states)):
                    prev_state = states[second_state]
                    sum += M[second_state, i - 1] * self.transitions[prev_state].get(states[state], 0) * self.emissions[
                        states[state]].get(sequence[i], 0)
                M[state, i] = sum
        return states[np.argmax(M[:, -1])], states[np.argmax(M[:, -1])] in safe_landings

    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.
    def viterbi(self, sequence):
        states = list(self.transitions.keys())
        states.remove("#")
        M = np.zeros((len(states), len(sequence)))
        path = np.zeros((len(states), len(sequence)), dtype=int)
        self.initialize_first_observation(sequence, states, M, path)

        for i in range(1, len(sequence)):
            for state in range(len(states)):
                max_prob = 0
                best_prev_state = 0
                for second_state in range(len(states)):
                    prev_state = states[second_state]
                    sum = M[second_state, i - 1] * self.transitions[prev_state].get(states[state], 0) * self.emissions[
                        states[state]].get(sequence[i], 0)
                    if sum > max_prob:
                        max_prob = sum
                        best_prev_state = second_state
                M[state, i] = max_prob
                path[state, i] = best_prev_state

        return self.backtrack_viterbi_path(path, states, np.argmax(M[:, -1]), len(sequence))

    def backtrack_viterbi_path(self, path, states, last_state, sequence_length):
        best_path = [last_state]
        for i in range(sequence_length - 1, 0, -1):
            last_state = path[last_state, i]
            best_path.insert(0, last_state)

        return [states[state_index] for state_index in best_path]

    def initialize_first_observation(self, sequence, states, M, path=None):
        state = 0
        while state < len(states):
            M[state, 0] = self.emissions[states[state]].get(sequence[0], 0) * self.transitions["#"].get(states[state],1.0)
            if path is not None:
                path[state, 0] = 0
            state += 1
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

def main():
    parser = argparse.ArgumentParser(description="HMM")
    parser.add_argument("basename", type=str, help="Basename")
    parser.add_argument("--generate", type=int, help="Generate")
    parser.add_argument("--forward", type=str, help="Forward")
    parser.add_argument("--viterbi", type=str, help="Viterbi")
    parser.add_argument("--output", type=str, help="Output")

    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.basename)

    if args.generate:
        generated = hmm.generate(args.generate)
        print("Generated sequence:")
        print("State sequence:", " ".join(generated.stateseq))
        print("Output sequence:", " ".join(generated.outputseq))

        if args.output:
            with open(args.output, 'w') as file:
                file.write(" ".join(generated.outputseq))
            print(f"Generated sequence saved at: {args.output}")

    if args.forward:
        with open(args.forward, 'r') as file:
            sequence = file.read().strip().split()

        final_state, is_safe = hmm.forward(sequence) if args.basename == "lander" else (hmm.forward(sequence), None)

        if args.basename == "lander":
            print(f"Final State: {final_state}")
            if is_safe:
                print("Landing is SAFE at the final state.")
            else:
                print("Landing is NOT SAFE at the final state.")
        else:
            print(f"Predicted final state using forward algorithm:: {final_state}")

    if args.viterbi:
        with open(args.viterbi, 'r') as file:
            sequence = file.read().strip().split()
            final_path = hmm.viterbi(sequence)
            print(f"Final Path: {final_path}")

if __name__ == "__main__":
    main()







