from pathlib import Path

import duckdb


DUCKLAKE_URL = (
    "ducklake:https://s3-cache.deadlock-api.com/"
    "fast/db_snapshot.ducklake"
)

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "matches.parquet"

# For the first prototype we only need a relatively small dataset.
MATCH_LIMIT = 20_000

# We already observed many valid matches with badge information
# in this range. Limiting the range also prevents us from scanning
# the entire hundreds-of-millions-row database during development.
MATCH_ID_MIN = 28_000_000
MATCH_ID_MAX = 29_000_000


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with duckdb.connect() as con:

        print("Connecting to Deadlock database...")

        con.execute("""
            INSTALL ducklake;
            LOAD ducklake;

            INSTALL httpfs;
            LOAD httpfs;

            CREATE OR REPLACE SECRET deadlock_s3 (
                TYPE S3,
                KEY_ID '',
                SECRET '',
                ENDPOINT 's3-cache.deadlock-api.com',
                URL_STYLE 'path',
                USE_SSL true
            );
        """)

        con.execute(
            f"ATTACH '{DUCKLAKE_URL}' AS db (READ_ONLY)"
        )

        con.execute("USE db.main")

        print("Connected.")
        print()
        print(f"Preparing up to {MATCH_LIMIT} matches...")

        query = f"""
            SELECT
                match_id,

                MIN(start_time) AS start_time,

                MIN(average_badge_team0)
                    AS badge_team0,

                MIN(average_badge_team1)
                    AS badge_team1,

                list(hero_id ORDER BY hero_id)
                    FILTER (WHERE team = 'Team0')
                    AS team0_heroes,

                list(hero_id ORDER BY hero_id)
                    FILTER (WHERE team = 'Team1')
                    AS team1_heroes,

                CASE
                    WHEN MIN(winning_team) = 'Team0'
                    THEN 1
                    ELSE 0
                END AS team0_won

            FROM match_player

            WHERE
                match_id >= {MATCH_ID_MIN}
                AND match_id < {MATCH_ID_MAX}

                AND game_mode = 'Normal'
                AND match_mode = 'Unranked'

                AND team IN (
                    'Team0',
                    'Team1'
                )

                AND winning_team IN (
                    'Team0',
                    'Team1'
                )

                AND average_badge_team0 > 0
                AND average_badge_team1 > 0

                AND hero_id IS NOT NULL

            GROUP BY match_id

            HAVING
                COUNT(*) = 12

                AND COUNT(*) FILTER (
                    WHERE team = 'Team0'
                ) = 6

                AND COUNT(*) FILTER (
                    WHERE team = 'Team1'
                ) = 6

                AND COUNT(hero_id) = 12

                AND MIN(average_badge_team0)
                    = MAX(average_badge_team0)

                AND MIN(average_badge_team1)
                    = MAX(average_badge_team1)

                AND MIN(winning_team)
                    = MAX(winning_team)

            LIMIT {MATCH_LIMIT}
        """

        output_path = OUTPUT_FILE.resolve().as_posix()

        print("Downloading dataset...")

        con.execute(f"""
            COPY (
                {query}
            )
            TO '{output_path}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
        """)

        print()
        print(f"Dataset saved to: {OUTPUT_FILE}")

    inspect_downloaded_dataset()


def inspect_downloaded_dataset():
    print()
    print("=" * 80)
    print("DOWNLOADED DATASET")
    print("=" * 80)

    with duckdb.connect() as con:
        total = con.execute("""
            SELECT COUNT(*)
            FROM read_parquet(?)
        """, [str(OUTPUT_FILE)]).fetchone()[0]

        print(f"Matches: {total}")

        print()
        print("Winner distribution:")

        winner_distribution = con.execute("""
            SELECT
                team0_won,
                COUNT(*) AS matches
            FROM read_parquet(?)
            GROUP BY team0_won
            ORDER BY team0_won
        """, [str(OUTPUT_FILE)]).fetchall()

        for team0_won, matches in winner_distribution:
            print(
                f"Team0 won = {team0_won}: "
                f"{matches} matches"
            )

        print()
        print("Sample:")

        sample = con.execute("""
    SELECT
        match_id,
        badge_team0,
        badge_team1,
        team0_heroes,
        team1_heroes,
        team0_won
    FROM read_parquet(?)
    LIMIT 10
""", [str(OUTPUT_FILE)]).fetchall()

        for row in sample:
            print(row)


if __name__ == "__main__":
    inspect_downloaded_dataset()