"""Synthetic, cross-domain datasets to prove the analytics engine is generic.

Each domain is a small star/snowflake of tables linked by shared keys, so the
profiler's value-overlap relationship discovery connects them with no domain
knowledge. The same engine answers "asset lifecycle", "slot recommendations",
"patient cost drivers", "student outcomes", and "listing pricing" because all
of those are just joins + aggregates + models over discovered relationships.

Every generator is seeded for reproducibility and writes CSVs to ``out_dir``,
returning ``{table_name: Path}``.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Callable
from pathlib import Path


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _pick(rng: random.Random, seq: list[str]) -> str:
    return seq[rng.randrange(len(seq))]


# ---------------------------------------------------------------------------
# Casino / gaming
# ---------------------------------------------------------------------------


def generate_casino(out_dir: Path, seed: int = 1, n_assets: int = 120, n_players: int = 400) -> dict[str, Path]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = [f"A{idx:04d}" for idx in range(1, n_assets + 1)]
    players = [f"P{idx:04d}" for idx in range(1, n_players + 1)]
    zones = ["Z01", "Z02", "Z03", "Z04"]
    denoms = [0.01, 0.05, 0.25, 1.0]
    games = ["MYSTIC KING", "COIN KINGDOM", "BUFFALO", "LIGHTNING LINK"]

    asset_rows: list[list] = []
    for a in assets:
        d = rng.choice(denoms)
        g = rng.choice(games)
        for day in range(1, 31):
            coin_in = max(0, int(rng.gauss(2000, 800) * (1 + d * 2)))
            net_win = int(coin_in * rng.uniform(-0.15, 0.12))
            asset_rows.append([f"2025-03-{day:02d}", a, coin_in, net_win, d, g, rng.choice(zones)])

    # Cover most assets so the profiler discovers a confident (>=0.8 coverage)
    # join from changeLog.assetId -> assetDaily.assetId (the event_impact anchor).
    change_rows: list[list] = []
    for a in assets[:110]:
        prev = _pick(rng, assets)
        df, dt = rng.choice(denoms), rng.choice(denoms)
        change_rows.append(
            [f"2025-03-{rng.randint(10, 25):02d}", "Denom Change", a, prev, df, dt,
             rng.choice(games), rng.choice(games)]
        )

    session_rows: list[list] = []
    for i in range(1500):
        p = _pick(rng, players)
        a = _pick(rng, assets)
        day = rng.randint(1, 30)
        coin_in = max(0, int(rng.gauss(300, 150)))
        session_rows.append([f"S{i:05d}", p, a, f"2025-03-{day:02d}", coin_in, rng.randint(5, 120)])

    player_rows: list[list] = []
    for p in players:
        tier = _pick(rng, ["Bronze", "Gold", "Platinum"])
        player_rows.append([p, tier, rng.randint(30, 2000), _pick(rng, ["NV", "CA", "AZ"])])

    paths = {
        "assetDaily": out_dir / "assetDaily.csv",
        "changeLog": out_dir / "changeLog.csv",
        "sessions": out_dir / "sessions.csv",
        "players": out_dir / "players.csv",
    }
    _write(paths["assetDaily"], ["day", "assetId", "coinIn", "netWin", "denom", "gameTitle", "zone"], asset_rows)
    _write(paths["changeLog"], ["day", "changeType", "assetId", "assetPrev", "denomFrom", "denomTo", "gameTitleFrom", "gameTitleTo"], change_rows)
    _write(paths["sessions"], ["sessionId", "playerId", "assetId", "day", "coinIn", "timeOnDevice"], session_rows)
    _write(paths["players"], ["playerId", "tier", "tenureDays", "state"], player_rows)
    return paths


# ---------------------------------------------------------------------------
# E-commerce
# ---------------------------------------------------------------------------


def generate_ecommerce(out_dir: Path, seed: int = 2, n_customers: int = 300, n_products: int = 80) -> dict[str, Path]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    customers = [f"C{idx:04d}" for idx in range(1, n_customers + 1)]
    products = [f"PR{idx:04d}" for idx in range(1, n_products + 1)]
    cities = ["Las Vegas", "Reno", "Phoenix", "LA"]
    segments = ["New", "Active", "VIP"]
    categories = ["Electronics", "Home", "Apparel", "Toys"]

    customer_rows: list[list] = []
    for c in customers:
        seg = _pick(rng, segments)
        # VIP customers have higher-value baskets (signal for modeling).
        customer_rows.append([c, _pick(rng, cities), seg, f"2024-{rng.randint(1,12):02d}-01"])

    product_rows: list[list] = []
    prices: dict[str, float] = {}
    for p in products:
        cat = _pick(rng, categories)
        price = round(rng.uniform(5, 500), 2)
        prices[p] = price
        product_rows.append([p, cat, price])

    order_rows: list[list] = []
    for i in range(2000):
        c = _pick(rng, customers)
        p = _pick(rng, products)
        seg = next(row[2] for row in customer_rows if row[0] == c)
        base = prices[p]
        qty = rng.randint(1, 5)
        amount = round(base * qty * (1.4 if seg == "VIP" else 1.0), 2)
        order_rows.append([f"O{i:05d}", c, p, f"2025-04-{rng.randint(1,30):02d}", amount, qty])

    event_rows: list[list] = []
    for i in range(3000):
        event_rows.append([f"E{i:05d}", _pick(rng, customers), _pick(rng, products),
                           f"2025-04-{rng.randint(1,30):02d}", _pick(rng, ["view", "cart", "purchase"])])

    paths = {
        "orders": out_dir / "orders.csv",
        "customers": out_dir / "customers.csv",
        "products": out_dir / "products.csv",
        "events": out_dir / "events.csv",
    }
    _write(paths["orders"], ["orderId", "customerId", "productId", "day", "amount", "quantity"], order_rows)
    _write(paths["customers"], ["customerId", "city", "segment", "signupDate"], customer_rows)
    _write(paths["products"], ["productId", "category", "price"], product_rows)
    _write(paths["events"], ["eventId", "customerId", "productId", "day", "type"], event_rows)
    return paths


# ---------------------------------------------------------------------------
# Healthcare
# ---------------------------------------------------------------------------


def generate_health(out_dir: Path, seed: int = 3, n_patients: int = 350, n_providers: int = 40) -> dict[str, Path]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patients = [f"PT{idx:04d}" for idx in range(1, n_patients + 1)]
    providers = [f"PV{idx:03d}" for idx in range(1, n_providers + 1)]
    regions = ["North", "South", "East", "West"]
    specialties = ["Cardiology", "Primary", "Ortho", "Oncology"]

    patient_rows: list[list] = []
    ages: dict[str, int] = {}
    for p in patients:
        age = rng.randint(18, 90)
        ages[p] = age
        patient_rows.append([p, age, _pick(rng, regions), f"2023-{rng.randint(1,12):02d}-01"])

    provider_rows: list[list] = []
    for pv in providers:
        provider_rows.append([pv, _pick(rng, specialties), _pick(rng, ["General", "Teaching"])])

    visit_rows: list[list] = []
    for i in range(1800):
        p = _pick(rng, patients)
        pv = _pick(rng, providers)
        # Cost rises with age (signal).
        cost = int(rng.gauss(400 + ages[p] * 8, 200))
        visit_rows.append([f"V{i:05d}", p, pv, f"2025-05-{rng.randint(1,30):02d}", max(50, cost), f"D{rng.randint(1,9)}"])

    lab_rows: list[list] = []
    for i in range(1200):
        p = _pick(rng, patients)
        lab_rows.append([f"L{i:05d}", p, f"2025-05-{rng.randint(1,30):02d}", round(rng.gauss(100, 30), 1)])

    paths = {
        "visits": out_dir / "visits.csv",
        "patients": out_dir / "patients.csv",
        "providers": out_dir / "providers.csv",
        "labs": out_dir / "labs.csv",
    }
    _write(paths["visits"], ["visitId", "patientId", "providerId", "day", "cost", "dxCode"], visit_rows)
    _write(paths["patients"], ["patientId", "age", "region", "enrolledDate"], patient_rows)
    _write(paths["providers"], ["providerId", "specialty", "hospital"], provider_rows)
    _write(paths["labs"], ["labId", "patientId", "day", "value"], lab_rows)
    return paths


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------


def generate_education(out_dir: Path, seed: int = 4, n_students: int = 400, n_courses: int = 50) -> dict[str, Path]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    students = [f"ST{idx:04d}" for idx in range(1, n_students + 1)]
    courses = [f"CR{idx:03d}" for idx in range(1, n_courses + 1)]
    schools = ["Lincoln", "Washington", "Jefferson"]
    subjects = ["Math", "Reading", "Science", "History"]
    levels = ["K5", "Middle", "High"]

    student_rows: list[list] = []
    cohorts: dict[str, int] = {}
    for s in students:
        cy = rng.randint(2015, 2024)
        cohorts[s] = cy
        student_rows.append([s, _pick(rng, schools), cy])

    course_rows: list[list] = []
    for c in courses:
        course_rows.append([c, _pick(rng, subjects), _pick(rng, levels)])

    enroll_rows: list[list] = []
    for i in range(2500):
        s = _pick(rng, students)
        c = _pick(rng, courses)
        grade = max(0.0, min(100.0, rng.gauss(70 + (2024 - cohorts[s]), 12)))
        enroll_rows.append([f"E{i:05d}", s, c, f"T{rng.randint(1,4)}", round(grade, 1), rng.randint(20, 120)])

    assess_rows: list[list] = []
    for i in range(2000):
        s = _pick(rng, students)
        c = _pick(rng, courses)
        score = max(0.0, min(100.0, rng.gauss(68 + (2024 - cohorts[s]) * 0.5, 15)))
        assess_rows.append([f"A{i:05d}", s, c, f"2025-06-{rng.randint(1,30):02d}", round(score, 1)])

    paths = {
        "enrollments": out_dir / "enrollments.csv",
        "students": out_dir / "students.csv",
        "courses": out_dir / "courses.csv",
        "assessments": out_dir / "assessments.csv",
    }
    _write(paths["enrollments"], ["enrollId", "studentId", "courseId", "term", "grade", "hours"], enroll_rows)
    _write(paths["students"], ["studentId", "school", "cohortYear"], student_rows)
    _write(paths["courses"], ["courseId", "subject", "level"], course_rows)
    _write(paths["assessments"], ["assessId", "studentId", "courseId", "day", "score"], assess_rows)
    return paths


# ---------------------------------------------------------------------------
# Real estate
# ---------------------------------------------------------------------------


def generate_realestate(out_dir: Path, seed: int = 5, n_agents: int = 60, n_listings: int = 500) -> dict[str, Path]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = [f"AG{idx:03d}" for idx in range(1, n_agents + 1)]
    listings = [f"LI{idx:04d}" for idx in range(1, n_listings + 1)]
    neighborhoods = [f"NB{idx:02d}" for idx in range(1, 25)]
    cities = ["Henderson", "Summerlin", "Las Vegas", "Mesquite"]
    offices = ["Office A", "Office B", "Office C"]

    agent_rows: list[list] = []
    for a in agents:
        agent_rows.append([a, _pick(rng, offices), rng.randint(1, 25)])

    nb_rows: list[list] = []
    incomes: dict[str, int] = {}
    for nb in neighborhoods:
        inc = rng.randint(40_000, 180_000)
        incomes[nb] = inc
        nb_rows.append([nb, _pick(rng, cities), inc])

    listing_rows: list[list] = []
    prices: dict[str, int] = {}
    for li in listings:
        nb = _pick(rng, neighborhoods)
        a = _pick(rng, agents)
        sqft = rng.randint(800, 4000)
        price = int((incomes[nb] / 1000) * sqft * rng.uniform(0.8, 1.3))
        prices[li] = price
        listing_rows.append([li, a, nb, f"2025-07-{rng.randint(1,30):02d}", price, sqft])

    txn_rows: list[list] = []
    for i in range(800):
        li = _pick(rng, listings)
        buyer = f"B{rng.randint(1,1000):04d}"
        sale = int(prices[li] * rng.uniform(0.9, 1.1))
        txn_rows.append([f"T{i:05d}", li, buyer, f"2025-07-{rng.randint(1,30):02d}", sale])

    paths = {
        "listings": out_dir / "listings.csv",
        "agents": out_dir / "agents.csv",
        "neighborhoods": out_dir / "neighborhoods.csv",
        "transactions": out_dir / "transactions.csv",
    }
    _write(paths["listings"], ["listingId", "agentId", "neighborhoodId", "day", "price", "sqft"], listing_rows)
    _write(paths["agents"], ["agentId", "office", "tenureYears"], agent_rows)
    _write(paths["neighborhoods"], ["neighborhoodId", "city", "medianIncome"], nb_rows)
    _write(paths["transactions"], ["txnId", "listingId", "buyerId", "day", "salePrice"], txn_rows)
    return paths


GENERATORS: dict[str, Callable[[Path], dict[str, Path]]] = {
    "casino": lambda d: generate_casino(d),
    "ecommerce": lambda d: generate_ecommerce(d),
    "health": lambda d: generate_health(d),
    "education": lambda d: generate_education(d),
    "realestate": lambda d: generate_realestate(d),
}


def generate_all(out_dir: Path, domains: list[str] | None = None) -> dict[str, dict[str, Path]]:
    """Generate every requested domain (default: all) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = domains or list(GENERATORS)
    return {name: GENERATORS[name](out_dir) for name in domains}
