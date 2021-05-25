STRATEGY_FOLDERS = [
    "exampleStrats",
    "valadaptive",
    "nekiwo",
    "edward",
    "misc",
    "saffron",
#    "aaaa-trsh",
#    "phoenix",
#    "l4vr0v",
#    "smough",
#    "dratini0",
#    "decxjo",
#    "Nobody5050",
#    "sanscipher"
]
CACHE_FILE = "cache.json"

# Minimum rounds to run
MIN_ROUNDS = 10

# Maximum Standard Deviation to keep calculating after MIN_ROUNDS
MIN_SD = 0.02

import multiprocessing
import os
import itertools
import importlib
import pathlib

import numpy as np
import random
from multiprocessing import Pool, cpu_count
import statistics
import argparse
import sys
import json

parser = argparse.ArgumentParser(description="Run the Prisoner's Dilemma simulation.")
parser.add_argument(
    "-n",
    "--num-runs",
    dest="num_runs",
    type=int,
    default=100,
    help="Number of runs to average out",
)

cacheparser = parser.add_argument_group("Cache")

cacheparser.add_argument(
    "--delete-cache",
    "--remove-cache",
    dest="delete_cache",
    action="store_true",
    default=False,
    help="Deletes the cache."
)

cacheparser.add_argument(
    "--cache-file",
    dest="cache_file",
    type=str,
    default="",
    help="Specifies the cache file to use."
)

parser.add_argument(
    "-j",
    "--num-processes",
    dest="processes",
    type=int,
    default=cpu_count(),
    help="Number of processes to run the simulation with. By default, this is the same as your CPU core count.",
)


args = parser.parse_args()


NUM_RUNS = args.num_runs

# The i-j-th element of this array is how many points you
# receive if you do play i, and your opponent does play j.
pointsArray = [
    [1,5],
    [0,3]
]

moveLabels = ["D", "C"]

def getVisibleHistory(history, player, turn):
    historySoFar = history[:,:turn].copy()
    if player == 1:
        historySoFar = np.flip(historySoFar,0)
    return historySoFar

def strategyMove(move):
    if type(move) is str:
        defects = ["defect","tell truth"]
        return 0 if (move in defects) else 1
    else:
        # Coerce all moves to be 0 or 1 so strategies can safely assume 0/1's only
        return int(bool(move))

def runRound(pair, moduleA, moduleB):
    memoryA = None
    memoryB = None

    LENGTH_OF_GAME = int(200-40*np.log(1-random.random())) # The games are a minimum of 200 turns long. The np.log here guarantees that every turn after the 200th has an equal (low) chance of being the final turn.
    history = np.zeros((2,LENGTH_OF_GAME),dtype=int)
    historyFlipped = np.zeros((2,LENGTH_OF_GAME),dtype=int)

    for turn in range(LENGTH_OF_GAME):
        playerAmove, memoryA = moduleA.strategy(history[:,:turn].copy(),memoryA)
        playerBmove, memoryB = moduleB.strategy(historyFlipped[:,:turn].copy(),memoryB)
        history[0, turn] = strategyMove(playerAmove)
        history[1, turn] = strategyMove(playerBmove)
        historyFlipped[0,turn] = history[1,turn]
        historyFlipped[1,turn] = history[0,turn]

    return history

def tallyRoundScores(history):
    scoreA = 0
    scoreB = 0
    ROUND_LENGTH = history.shape[1]
    for turn in range(ROUND_LENGTH):
        playerAmove = history[0,turn]
        playerBmove = history[1,turn]
        scoreA += pointsArray[playerAmove][playerBmove]
        scoreB += pointsArray[playerBmove][playerAmove]
    return scoreA/ROUND_LENGTH, scoreB/ROUND_LENGTH

def pad(stri, leng):
    result = stri
    for i in range(len(stri),leng):
        result = result+" "
    return result

def progressBar(width, completion):
    numCompleted = round(width * completion)
    return f"[{'=' * numCompleted}{' ' * (width - numCompleted)}]"

def runRounds(pair):
    moduleA = importlib.import_module(pair[0][0])
    moduleB = importlib.import_module(pair[1][0])

    allScoresA = []
    allScoresB = []
    firstRoundHistory = None
    for i in range(NUM_RUNS):
        roundHistory = runRound(pair, moduleA, moduleB)
        scoresA, scoresB = tallyRoundScores(roundHistory)
        if i == 0:
            firstRoundHistory = roundHistory
            if scoresA == 3 and scoresB == 3:
                return [pair,[3, 3, 0, 0, firstRoundHistory.tolist()]]
            if scoresA == 0:
                return [pair,[0, 5, 0, 0, firstRoundHistory.tolist()]]
            if scoresB == 0:
                return [pair,[5, 0, 0, 0, firstRoundHistory.tolist()]]

        allScoresA.append(scoresA)
        allScoresB.append(scoresB)
        if i == MIN_ROUNDS and MIN_ROUNDS > 0:
            if statistics.stdev(allScoresA) < MIN_SD or statistics.stdev(allScoresB) < MIN_SD:
                avgScoreA = statistics.mean(allScoresA)
                avgScoreB = statistics.mean(allScoresB)
                stdevA = statistics.stdev(allScoresA) if len(allScoresA) > 1 else 0
                stdevB = statistics.stdev(allScoresB) if len(allScoresB) > 1 else 0

                return [pair,[avgScoreA, avgScoreB, stdevA, stdevB, firstRoundHistory.tolist()]]

    avgScoreA = statistics.mean(allScoresA)
    avgScoreB = statistics.mean(allScoresB)
    stdevA = statistics.stdev(allScoresA) if len(allScoresA) > 1 else 0
    stdevB = statistics.stdev(allScoresB) if len(allScoresB) > 1 else 0

    return [pair,[avgScoreA, avgScoreB, stdevA, stdevB, firstRoundHistory.tolist()]]

def pool_init(l):
    global lock
    lock = l

def loadCache():
    if args.delete_cache:
        return {}
    try:
        with open(CACHE_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def progressBar(completion):
    numCompleted = round(50 * completion)
    return f"[{'=' * numCompleted}{' ' * (50 - numCompleted)}]"


def runFullPairingTournament(inFolders, outFile):
    print("Starting tournament, reading files from " + ", ".join(inFolders))

    cache = loadCache()

    STRATEGY_LIST = []
    for inFolder in inFolders:
        for file in os.listdir(inFolder):
            if file.endswith(".py"):
                STRATEGY_LIST.append([f"{inFolder}.{file[:-3]}",pathlib.Path(f"{inFolder}/{file[:-3]}.py").stat().st_mtime_ns])

    if len(STRATEGY_LIST) < 2:
        raise ValueError('Not enough strategies!')

    combinations = list(itertools.combinations(STRATEGY_LIST, r=2))

    for s in STRATEGY_LIST:
        combinations.append([s,s])

    numCombinations = len(combinations)

    sys.stdout.write(f"\r{0}/{numCombinations} pairings ({NUM_RUNS} runs per pairing, 0 hits, 0 misses) {progressBar(0)}")
    sys.stdout.flush()

    i = len(combinations)

    while i > 0:
        i -= 1
        if str(combinations[i][0][0])+","+str(combinations[i][0][1]) in cache:
            if str(combinations[i][1][0])+","+str(combinations[i][1][1]) in cache[str(combinations[i][0][0])+","+str(combinations[i][0][1])]:
                combinations.pop(i)
                continue

        if str(combinations[i][1][0])+","+str(combinations[i][1][1]) in cache:
            if str(combinations[i][0][0])+","+str(combinations[i][0][1]) in cache[str(combinations[i][1][0])+","+str(combinations[i][1][1])]:
                combinations.pop(i)

    skippedCombinations = numCombinations-len(combinations)

    sys.stdout.write(f"\r{skippedCombinations}/{numCombinations} pairings ({NUM_RUNS} runs per pairing, {skippedCombinations} hits, {numCombinations-skippedCombinations} misses) {progressBar(0)}")
    sys.stdout.flush()

    progressCounter = 0

    #with Pool(args.processes, initializer=pool_init, initargs=(multiprocessing.Lock(),)) as p:
    with Pool(args.processes) as p:
        for v in p.imap(runRounds, combinations):
            progressCounter += 1
            sys.stdout.write(f"\r{skippedCombinations+progressCounter}/{numCombinations} pairings ({NUM_RUNS} runs per pairing, {skippedCombinations} hits, {numCombinations-skippedCombinations} misses) {progressBar(progressCounter/(numCombinations-skippedCombinations))}")
            sys.stdout.flush()
            if str(v[0][0][0])+","+str(v[0][0][1]) not in cache:
                cache[str(v[0][0][0])+","+str(v[0][0][1])] = {}
            cache[str(v[0][0][0])+","+str(v[0][0][1])][str(v[0][1][0])+","+str(v[0][1][1])] = v[1]

    with open(outFile, 'w') as of:
        json.dump(cache, of)

    print("\nDone with everything! Results file written to "+outFile)

if __name__ == "__main__":
    runFullPairingTournament(STRATEGY_FOLDERS, CACHE_FILE)
