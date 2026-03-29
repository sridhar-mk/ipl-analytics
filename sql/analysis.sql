-- ============================================================
-- IPL Player Performance Analytics — SQL Analysis
-- Database: ipl.db (SQLite)
-- Run after data_cleaning.py
-- ============================================================


-- ─────────────────────────────────────────
-- SECTION 1: SEASON OVERVIEW
-- ─────────────────────────────────────────

-- Q1: Total matches, runs and sixes per season
SELECT
    season,
    COUNT(DISTINCT match_id)   AS total_matches,
    SUM(total_runs)            AS total_runs,
    SUM(is_six)                AS total_sixes,
    ROUND(AVG(total_runs), 2)  AS avg_runs_per_ball
FROM deliveries
GROUP BY season
ORDER BY season;


-- Q2: IPL title winners and how many times each team won
SELECT
    winner                    AS team,
    COUNT(*)                  AS titles
FROM matches
WHERE match_type = 'Final'
  AND winner IS NOT NULL
GROUP BY winner
ORDER BY titles DESC;


-- Q3: Season-wise highest team score in a single innings
SELECT
    season,
    batting_team              AS team,
    match_id,
    SUM(total_runs)           AS innings_total
FROM deliveries
WHERE inning = 1
GROUP BY season, batting_team, match_id
ORDER BY innings_total DESC
LIMIT 10;


-- ─────────────────────────────────────────
-- SECTION 2: BATTING ANALYSIS
-- ─────────────────────────────────────────

-- Q4: All-time top 15 run scorers (min 20 matches)
SELECT
    player,
    SUM(matches)              AS total_matches,
    SUM(runs)                 AS total_runs,
    SUM(balls_faced)          AS total_balls,
    ROUND(SUM(runs) * 100.0
          / NULLIF(SUM(balls_faced),0), 2)  AS career_strike_rate,
    ROUND(SUM(runs) * 1.0
          / NULLIF(SUM(times_dismissed),0), 2) AS batting_avg,
    SUM(sixes)                AS total_sixes,
    SUM(fours)                AS total_fours
FROM batting_summary
GROUP BY player
HAVING total_matches >= 20
ORDER BY total_runs DESC
LIMIT 15;


-- Q5: Top run scorers per season using RANK() window function
WITH season_runs AS (
    SELECT
        player,
        season,
        SUM(runs) AS runs
    FROM batting_summary
    GROUP BY player, season
),
ranked AS (
    SELECT
        player,
        season,
        runs,
        RANK() OVER (PARTITION BY season ORDER BY runs DESC) AS season_rank
    FROM season_runs
)
SELECT * FROM ranked
WHERE season_rank <= 3
ORDER BY season, season_rank;


-- Q6: Most consistent batsmen — low variance in runs across seasons
-- (players who score well every season, not just once)
WITH season_avgs AS (
    SELECT
        player,
        season,
        SUM(runs) AS season_runs
    FROM batting_summary
    GROUP BY player, season
),
consistency AS (
    SELECT
        player,
        COUNT(season)                          AS seasons_played,
        ROUND(AVG(season_runs), 1)             AS avg_runs_per_season,
        ROUND(MIN(season_runs), 1)             AS min_season_runs,
        ROUND(MAX(season_runs), 1)             AS max_season_runs,
        ROUND(MAX(season_runs)
              - MIN(season_runs), 1)           AS variance_range
    FROM season_avgs
    GROUP BY player
    HAVING seasons_played >= 5
)
SELECT * FROM consistency
ORDER BY avg_runs_per_season DESC
LIMIT 15;


-- Q7: Best batting strike rate in death overs (overs 17-20, min 100 balls)
SELECT
    batter                                          AS player,
    COUNT(*)                                        AS balls_faced,
    SUM(batsman_runs)                               AS runs,
    SUM(is_six)                                     AS sixes,
    ROUND(SUM(batsman_runs)*100.0/COUNT(*), 2)      AS death_strike_rate
FROM deliveries
WHERE over >= 16
  AND extras_type NOT IN ('wides','noballs')
GROUP BY batter
HAVING balls_faced >= 100
ORDER BY death_strike_rate DESC
LIMIT 10;


-- Q8: Batsmen with most sixes all time
SELECT
    batter          AS player,
    SUM(is_six)     AS total_sixes,
    SUM(is_four)    AS total_fours,
    COUNT(DISTINCT match_id) AS matches
FROM deliveries
GROUP BY batter
HAVING matches >= 20
ORDER BY total_sixes DESC
LIMIT 10;


-- Q9: Running total of runs per batsman across seasons (career progression)
WITH career AS (
    SELECT
        player,
        season,
        SUM(runs) AS season_runs
    FROM batting_summary
    GROUP BY player, season
)
SELECT
    player,
    season,
    season_runs,
    SUM(season_runs) OVER (
        PARTITION BY player
        ORDER BY season
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_runs
FROM career
WHERE player IN ('V Kohli','RG Sharma','SK Raina','CH Gayle','DA Warner')
ORDER BY player, season;


-- ─────────────────────────────────────────
-- SECTION 3: BOWLING ANALYSIS
-- ─────────────────────────────────────────

-- Q10: All-time top wicket takers (min 20 matches)
SELECT
    player,
    SUM(matches)              AS total_matches,
    SUM(wickets)              AS total_wickets,
    ROUND(SUM(runs_conceded) * 6.0
          / NULLIF(SUM(legal_balls),0), 2)   AS career_economy,
    ROUND(SUM(legal_balls) * 1.0
          / NULLIF(SUM(wickets),0), 2)        AS bowling_strike_rate,
    ROUND(SUM(runs_conceded) * 1.0
          / NULLIF(SUM(wickets),0), 2)        AS bowling_avg
FROM bowling_summary
GROUP BY player
HAVING total_matches >= 20
ORDER BY total_wickets DESC
LIMIT 15;


-- Q11: Best death over bowlers (overs 17-20, min 200 legal balls)
SELECT
    bowler                                              AS player,
    SUM(is_legal_delivery)                              AS balls_bowled,
    SUM(is_wicket)                                      AS wickets,
    SUM(total_runs)                                     AS runs_given,
    ROUND(SUM(total_runs)*6.0
          / NULLIF(SUM(is_legal_delivery),0), 2)        AS death_economy
FROM deliveries
WHERE over >= 16
GROUP BY bowler
HAVING balls_bowled >= 200
ORDER BY death_economy ASC
LIMIT 10;


-- Q12: Best powerplay bowlers (overs 1-6, min 150 legal balls)
SELECT
    bowler                                              AS player,
    SUM(is_legal_delivery)                              AS balls_bowled,
    SUM(is_wicket)                                      AS wickets,
    ROUND(SUM(total_runs)*6.0
          / NULLIF(SUM(is_legal_delivery),0), 2)        AS powerplay_economy
FROM deliveries
WHERE over < 6
GROUP BY bowler
HAVING balls_bowled >= 150
ORDER BY powerplay_economy ASC
LIMIT 10;


-- Q13: Bowler vs batsman head-to-head (min 12 balls)
SELECT
    bowler,
    batter,
    COUNT(*)                                        AS balls,
    SUM(batsman_runs)                               AS runs_given,
    SUM(is_wicket)                                  AS dismissals,
    ROUND(SUM(batsman_runs)*100.0/COUNT(*), 2)      AS strike_rate_allowed
FROM deliveries
GROUP BY bowler, batter
HAVING balls >= 12
ORDER BY dismissals DESC, strike_rate_allowed ASC
LIMIT 20;


-- ─────────────────────────────────────────
-- SECTION 4: TEAM & MATCH ANALYSIS
-- ─────────────────────────────────────────

-- Q14: Team win percentage all time
SELECT
    team,
    total_matches,
    wins,
    ROUND(wins * 100.0 / total_matches, 1)  AS win_pct
FROM (
    SELECT
        team1 AS team,
        COUNT(*) AS total_matches,
        SUM(CASE WHEN winner = team1 THEN 1 ELSE 0 END) AS wins
    FROM matches
    GROUP BY team1
    UNION ALL
    SELECT
        team2 AS team,
        COUNT(*) AS total_matches,
        SUM(CASE WHEN winner = team2 THEN 1 ELSE 0 END) AS wins
    FROM matches
    GROUP BY team2
) sub
GROUP BY team
HAVING SUM(total_matches) >= 30
ORDER BY win_pct DESC;


-- Q15: Toss impact — does winning toss help?
SELECT
    toss_decision,
    COUNT(*)                                            AS matches,
    SUM(toss_winner_won)                                AS toss_winner_won,
    ROUND(SUM(toss_winner_won)*100.0/COUNT(*), 1)       AS win_pct
FROM matches
GROUP BY toss_decision;


-- Q16: Venue-wise average first innings score and win % batting first
SELECT
    m.venue,
    COUNT(DISTINCT m.id)                               AS matches_played,
    ROUND(AVG(inn.innings_total), 1)                   AS avg_first_innings_score,
    SUM(CASE WHEN m.toss_decision='bat'
             AND m.toss_winner=m.winner THEN 1 ELSE 0 END) AS bat_first_wins
FROM matches m
JOIN (
    SELECT match_id, SUM(total_runs) AS innings_total
    FROM deliveries WHERE inning=1
    GROUP BY match_id
) inn ON inn.match_id = m.id
GROUP BY m.venue
HAVING matches_played >= 10
ORDER BY avg_first_innings_score DESC
LIMIT 15;


-- Q17: Head to head record between any two teams using CTE
WITH h2h AS (
    SELECT
        team1, team2, winner,
        CASE WHEN winner = team1 THEN 1 ELSE 0 END AS team1_won
    FROM matches
    WHERE (team1='Mumbai Indians' AND team2='Chennai Super Kings')
       OR (team1='Chennai Super Kings' AND team2='Mumbai Indians')
)
SELECT
    'Mumbai Indians'   AS team,
    SUM(CASE WHEN winner='Mumbai Indians' THEN 1 ELSE 0 END)    AS wins,
    COUNT(*)                                                      AS total_matches,
    ROUND(SUM(CASE WHEN winner='Mumbai Indians' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS win_pct
FROM h2h
UNION ALL
SELECT
    'Chennai Super Kings',
    SUM(CASE WHEN winner='Chennai Super Kings' THEN 1 ELSE 0 END),
    COUNT(*),
    ROUND(SUM(CASE WHEN winner='Chennai Super Kings' THEN 1 ELSE 0 END)*100.0/COUNT(*),1)
FROM h2h;


-- Q18: Player of the match leaders — most impactful players
SELECT
    player_of_match        AS player,
    COUNT(*)               AS awards,
    COUNT(DISTINCT season) AS seasons_active
FROM matches
WHERE player_of_match IS NOT NULL
GROUP BY player_of_match
ORDER BY awards DESC
LIMIT 15;


-- Q19: Biggest winning margins by runs and wickets
SELECT 'By Runs' AS type, winner, result_margin AS margin, season, city
FROM matches WHERE result='runs'
ORDER BY result_margin DESC LIMIT 5;

SELECT 'By Wickets' AS type, winner, result_margin AS margin, season, city
FROM matches WHERE result='wickets'
ORDER BY result_margin DESC LIMIT 5;


-- Q20: Super over matches — how many and who won most
SELECT
    winner,
    COUNT(*) AS super_over_wins
FROM matches
WHERE super_over = 'Y'
GROUP BY winner
ORDER BY super_over_wins DESC;


-- ─────────────────────────────────────────
-- SECTION 5: PLAYER AUCTION VALUE SCORE
-- Our original metric — nobody else has this
-- ─────────────────────────────────────────

-- Q21: Custom Player Auction Value Score
-- Combines: runs, strike rate, consistency, sixes, match impact
-- Scale: 0–100. Higher = better auction pick.
WITH batting AS (
    SELECT
        player,
        SUM(matches)        AS matches,
        SUM(runs)           AS total_runs,
        SUM(balls_faced)    AS total_balls,
        SUM(sixes)          AS total_sixes,
        ROUND(SUM(runs)*100.0/NULLIF(SUM(balls_faced),0),2) AS strike_rate,
        ROUND(SUM(runs)*1.0/NULLIF(SUM(times_dismissed),0),2) AS avg,
        COUNT(DISTINCT season) AS seasons
    FROM batting_summary
    GROUP BY player
    HAVING matches >= 15
),
pom AS (
    SELECT player_of_match AS player, COUNT(*) AS awards
    FROM matches WHERE player_of_match IS NOT NULL
    GROUP BY player_of_match
),
combined AS (
    SELECT
        b.player,
        b.matches,
        b.total_runs,
        b.strike_rate,
        b.avg,
        b.total_sixes,
        b.seasons,
        COALESCE(p.awards, 0) AS awards,
        -- Auction Value Score formula
        ROUND(
            (b.total_runs    / 50.0)   * 30   -- runs component (30%)
          + (b.strike_rate   / 200.0)  * 25   -- strike rate (25%)
          + (b.avg           / 60.0)   * 20   -- average (20%)
          + (b.total_sixes   / 20.0)   * 15   -- six-hitting (15%)
          + (COALESCE(p.awards,0)/5.0) * 10   -- match impact (10%)
        , 1) AS auction_value_score
    FROM batting b
    LEFT JOIN pom p ON p.player = b.player
)
SELECT
    player,
    matches,
    total_runs,
    strike_rate,
    avg,
    total_sixes,
    awards         AS player_of_match_awards,
    auction_value_score
FROM combined
ORDER BY auction_value_score DESC
LIMIT 20;
