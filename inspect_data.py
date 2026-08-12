import duckdb

DUCKLAKE_URL = "ducklake:https://s3-cache.deadlock-api.com/fast/db_snapshot.ducklake"


with duckdb.connect() as con:
    # ---------------------------------------------------------
    # 1. Load extensions needed for remote database access
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 2. Attach Deadlock database
    # ---------------------------------------------------------

    con.execute(
        f"ATTACH '{DUCKLAKE_URL}' AS db (READ_ONLY)"
    )

    con.execute("USE db.main")


    # ---------------------------------------------------------
    # 3. Sample of potentially useful columns
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SAMPLE OF IMPORTANT COLUMNS")
    print("=" * 80)

    con.sql("""
        SELECT
            match_id,
            account_id,
            team,
            hero_id,
            party,
            player_rank_initial_display_rank,
            player_rank_initial_flat_progress,
            average_badge,
            average_badge_team0,
            average_badge_team1,
            winning_team,
            game_mode,
            match_mode,
            ranked_type
        FROM match_player
        LIMIT 30
    """).show(max_rows=30)


    # ---------------------------------------------------------
    # 4. Distinct categorical values
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DISTINCT VALUES")
    print("=" * 80)

    columns = [
        "team",
        "winning_team",
        "game_mode",
        "match_mode",
        "ranked_type"
    ]

    for column in columns:
        print(f"\n--- {column} ---")

        con.sql(f"""
            SELECT DISTINCT {column}
            FROM match_player
            ORDER BY {column}
            LIMIT 50
        """).show(max_rows=50)


    # ---------------------------------------------------------
    # 5. Rank sample
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RANK SAMPLE")
    print("=" * 80)

    con.sql("""
        SELECT
            match_id,
            account_id,
            player_rank_initial_display_rank,
            player_rank_initial_flat_progress,
            average_badge
        FROM match_player
        WHERE player_rank_initial_display_rank IS NOT NULL
        LIMIT 50
    """).show(max_rows=50)


    # ---------------------------------------------------------
    # 6. Rank null statistics
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RANK NULL STATISTICS")
    print("=" * 80)

    con.sql("""
        SELECT
            COUNT(*) AS total_rows,

            COUNT(player_rank_initial_display_rank)
                AS rows_with_display_rank,

            COUNT(player_rank_initial_flat_progress)
                AS rows_with_flat_progress,

            COUNT(average_badge)
                AS rows_with_average_badge

        FROM match_player
    """).show()


    # ---------------------------------------------------------
    # 7. Search all tables for region/server related columns
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("REGION / SERVER COLUMNS")
    print("=" * 80)

    rows = con.execute("""
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE
            lower(column_name) LIKE '%region%'
            OR lower(column_name) LIKE '%server%'
            OR lower(column_name) LIKE '%cluster%'
        ORDER BY table_name, column_name
    """).fetchall()

    if len(rows) == 0:
        print("No region/server/cluster columns found.")
    else:
        for row in rows:
            print(
                f"{row[0]}.{row[1]} - {row[2]}"
            )


    # ---------------------------------------------------------
    # 8. Check how many actually useful rank/badge values exist
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("USEFUL RANK / BADGE STATISTICS")
    print("=" * 80)

    con.sql("""
        SELECT
            COUNT(*) AS total_rows,

            COUNT(*) FILTER (
                WHERE player_rank_initial_display_rank > 0
            ) AS positive_display_rank,

            COUNT(*) FILTER (
                WHERE player_rank_initial_flat_progress > 0
            ) AS positive_flat_progress,

            COUNT(*) FILTER (
                WHERE average_badge > 0
            ) AS positive_average_badge,

            COUNT(*) FILTER (
                WHERE average_badge_team0 > 0
                AND average_badge_team1 > 0
            ) AS rows_with_both_team_badges

        FROM match_player
    """).show()


    # ---------------------------------------------------------
    # 9. Sample matches that have badge data for both teams
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BADGE SAMPLE")
    print("=" * 80)

    con.sql("""
        SELECT DISTINCT
            match_id,
            average_badge_team0,
            average_badge_team1,
            winning_team,
            match_mode,
            game_mode
        FROM match_player
        WHERE
            average_badge_team0 > 0
            AND average_badge_team1 > 0
            AND winning_team IN ('Team0', 'Team1')
        LIMIT 50
    """).show(max_rows=50)


    # ---------------------------------------------------------
    # 10. Number of matches by match mode
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("MATCH COUNTS")
    print("=" * 80)

    con.sql("""
        SELECT
            match_mode,
            COUNT(DISTINCT match_id) AS matches
        FROM match_player
        WHERE winning_team IN ('Team0', 'Team1')
        GROUP BY match_mode
        ORDER BY matches DESC
    """).show()


    # ---------------------------------------------------------
    # 11. Check whether matches really contain 12 players
    #     (6 players on each team)
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("COMPLETE MATCH CHECK")
    print("=" * 80)

    con.sql("""
        SELECT
            match_id,
            COUNT(*) AS player_count,

            COUNT(*) FILTER (
                WHERE team = 'Team0'
            ) AS team0_players,

            COUNT(*) FILTER (
                WHERE team = 'Team1'
            ) AS team1_players,

            MIN(average_badge_team0) AS badge_team0,
            MIN(average_badge_team1) AS badge_team1,
            MIN(winning_team) AS winner

        FROM match_player

        WHERE
            game_mode = 'Normal'
            AND winning_team IN ('Team0', 'Team1')

        GROUP BY match_id

        HAVING
            COUNT(*) = 12
            AND COUNT(*) FILTER (
                WHERE team = 'Team0'
            ) = 6
            AND COUNT(*) FILTER (
                WHERE team = 'Team1'
            ) = 6

        LIMIT 20
    """).show(max_rows=20)