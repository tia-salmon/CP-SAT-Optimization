"""Configuration file for values used in issue prioritization code."""

from pathlib import Path 
import dotenv
import os

dotenv.load_dotenv()

DB_USER = os.getenv("USER")
DB_PW = os.getenv("PASSWORD")
DB_HOST = "localhost"

VARYING_ELI = [0, 25, 50, 100]
VARYING_CONST = ["c1", "c1p1", "c1t1", "c1p1t1"]
VARYING_WEIGHT = ["c1p1t1", "c2p1t1", "c1p2t1", "c1p1t2"]
VARYING_ERROR_RATE = [0, 10, 20]
VARYING_LOCAL_CONST = ["cl", "clpl", "cltl", "clpltl"]

POPULATION_SIZE = 10 #10
AVG_RUN_SIZE = 15 # 15 # number of times to run each config
CARRY_OVER_PERCENTAGE = 10 
FINAL_RESULTS = Path("results")
PROJECTS = [4, 12, 13, 25]
