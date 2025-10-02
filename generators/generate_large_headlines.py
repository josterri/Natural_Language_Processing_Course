"""
Generate large news headlines dataset for advanced NLP tasks
Creates 25,000 headlines across 4 categories for embeddings and language modeling
"""

import pandas as pd
import random
import re
from datetime import datetime

# Set seed for reproducibility
random.seed(42)

# POLITICS TEMPLATES (50 templates)
POLITICS_TEMPLATES = [
    "{leader} announces new {policy} initiative",
    "{country} and {country2} sign historic {agreement}",
    "Parliament debates controversial {topic} bill",
    "{party} party wins majority in {location} election",
    "President {name} addresses concerns over {issue}",
    "New {policy} law passes despite opposition",
    "{country} faces criticism over {issue} policy",
    "Government unveils {number} billion {topic} plan",
    "{leader} meets with {leader2} to discuss {topic}",
    "Senate votes on {policy} reform bill",
    "{party} coalition faces challenges on {topic}",
    "Mayor {name} proposes {policy} changes",
    "{country} strengthens ties with {country2}",
    "Opposition criticizes government {policy} approach",
    "{leader} calls for unity on {topic} issue",
    "{country} imposes new sanctions on {country2}",
    "Lawmakers approve {number} billion {policy} package",
    "{leader} resigns amid {topic} scandal",
    "Voters reject {party} proposal on {issue}",
    "{country} reopens {agreement} negotiations with {country2}",
    "Governor {name} declares state of emergency over {issue}",
    "Political crisis deepens as {party} loses support",
    "{leader} vetoes controversial {policy} legislation",
    "International summit addresses global {topic} concerns",
    "{country} withdraws from {agreement} treaty",
    "Election results show shift toward {party} policies",
    "{leader} promises action on {issue} crisis",
    "Diplomatic tensions rise between {country} and {country2}",
    "Ministers clash over proposed {policy} reforms",
    "{party} launches campaign for {topic} reform",
    "Court rules against government {policy} plan",
    "{leader} faces impeachment over {issue} allegations",
    "Regional leaders meet to coordinate {topic} response",
    "{country} seeks international support for {issue} position",
    "Protests erupt after {party} pushes {policy} bill",
    "{leader} announces bid for {location} governorship",
    "Budget committee approves {number} million for {topic}",
    "Political analysts predict {party} victory in {location}",
    "{country} and {country2} reach compromise on {issue}",
    "Cabinet reshuffle puts focus on {policy} priorities",
    "{leader} condemns recent {issue} incidents",
    "Bipartisan group proposes {topic} solution",
    "{country} hosts peace talks between rival factions",
    "Constitutional amendment on {policy} moves forward",
    "Foreign minister discusses {agreement} with {country2}",
    "Local referendum on {issue} scheduled for {month}",
    "{leader} inaugurated with promise to fix {topic}",
    "Emergency session called to address {issue} situation",
    "{party} leadership contest focuses on {policy} direction",
    "International observers monitor {location} election process",
]

# SPORTS TEMPLATES (50 templates)
SPORTS_TEMPLATES = [
    "{team} defeats {team2} in thrilling {number} to {number2} victory",
    "{player} scores {number} points in championship game",
    "{team} wins {tournament} for {ordinal} consecutive year",
    "{player} breaks record with {number} {metric} performance",
    "{team} secures playoff spot with win over {team2}",
    "{sport} season kicks off with {team} vs {team2}",
    "{player} named {award} player of the month",
    "{team} coach discusses strategy after {result} loss",
    "{country} takes gold in {sport} at championships",
    "{player} injured in {team} game against {team2}",
    "{team} announces signing of star player {player}",
    "{tournament} final set between {team} and {team2}",
    "{player} leads {team} to {ordinal} straight victory",
    "{sport} fans celebrate as {team} wins title",
    "{team} trades {player} to {team2} in surprise move",
    "Underdog {team} stuns {team2} in upset victory",
    "{player} retires after {number} years with {team}",
    "{team} dominates {team2} with {number} point margin",
    "Record crowd watches {team} take on {team2}",
    "{player} suspended for {number} games after incident",
    "{team} clinches division title with dramatic win",
    "Controversy erupts as {team} protests referee decision",
    "{player} achieves career milestone of {number} {metric}",
    "{team} rebuilding begins after {result} season",
    "Rivals {team} and {team2} prepare for heated matchup",
    "{player} signs {number} million contract extension",
    "{team} suffers setback as key player {player} out for season",
    "Playoff picture becomes clearer after {team} victory",
    "{sport} legend {player} inducted into hall of fame",
    "{team} unveils new stadium for upcoming season",
    "{player} traded from {team} to {team2} in blockbuster deal",
    "Overtime thriller sees {team} edge past {team2}",
    "{tournament} bracket set as {team} earns top seed",
    "{team} coach fired after disappointing {result} finish",
    "{player} makes triumphant return for {team}",
    "Youth movement pays off as {team} advances",
    "{team} dynasty continues with another {tournament} win",
    "{player} controversy overshadows {team} success",
    "All-star {player} selected for {tournament} squad",
    "{team} home winning streak reaches {number} games",
    "Rivalry renewed as {team} faces {team2} again",
    "{player} breaks {sport} scoring record",
    "{team} medical staff clears {player} to play",
    "Draft pick {player} impresses in debut for {team}",
    "{team} makes history with {ordinal} title",
    "{player} dedicates victory to hometown fans",
    "Weather delays {team} versus {team2} matchup",
    "{tournament} favorites {team} stumble against {team2}",
    "{player} withdraws from {tournament} due to injury",
    "{sport} officials investigate {team} for violations",
]

# TECHNOLOGY TEMPLATES (50 templates)
TECHNOLOGY_TEMPLATES = [
    "{company} launches new {product} with {feature} feature",
    "{product} update brings {feature} to millions of users",
    "New study shows {technology} improves {metric} by {number} percent",
    "{company} announces {technology} breakthrough",
    "{product} sales reach {number} million in first quarter",
    "Researchers develop {technology} for {application}",
    "{company} releases {product} version {number} today",
    "{technology} market expected to grow {number} percent",
    "{company} faces criticism over {product} privacy concerns",
    "AI powered {product} transforms {industry} sector",
    "{company} partners with {company2} on {technology} project",
    "New {product} features {technology} and enhanced {feature}",
    "{technology} adoption increases among {demographic} users",
    "{company} unveils plans for {product} expansion",
    "Experts predict {technology} will revolutionize {industry}",
    "Cybersecurity breach affects {number} million {product} accounts",
    "{company} acquires {company2} for {number} billion",
    "Open source {technology} gains momentum in {industry}",
    "{product} battery life extended by {number} percent",
    "{company} stock surges after {product} announcement",
    "Developers create innovative {application} using {technology}",
    "Global {technology} conference showcases latest innovations",
    "{company} delays {product} launch due to technical issues",
    "Patent dispute between {company} and {company2} intensifies",
    "{technology} regulation proposed by government officials",
    "{product} compatibility issues frustrate {demographic} users",
    "{company} invests {number} million in {technology} research",
    "Viral {product} app reaches {number} million downloads",
    "Legacy {product} support ending after {number} years",
    "{company} CEO discusses {technology} vision at summit",
    "Benchmark tests reveal {product} outperforms competitors",
    "{technology} startups attract record venture capital",
    "{company} addresses {product} security vulnerabilities",
    "Next generation {technology} promises {number}x performance",
    "{product} ecosystem expands with third-party {feature}",
    "Industry analysts skeptical of {company} {technology} claims",
    "{company} opens new {technology} development center",
    "Hackers exploit {product} flaw affecting {number} devices",
    "{technology} standards body approves new specifications",
    "{company} discontinues unpopular {product} line",
    "Beta testers report positive feedback on {product}",
    "{technology} integration challenges delay {product} rollout",
    "{company} commits to carbon neutral {product} manufacturing",
    "Augmented reality {product} debuts in {number} markets",
    "{technology} chips face global supply shortage",
    "{company} settles antitrust lawsuit over {product} practices",
    "Quantum {technology} achieves new milestone",
    "{product} subscriptions top {number} million worldwide",
    "{company} rival {company2} challenges market dominance",
    "Edge computing transforms {industry} with {technology}",
]

# ENTERTAINMENT TEMPLATES (50 templates)
ENTERTAINMENT_TEMPLATES = [
    "{movie} tops box office with {number} million opening",
    "{celebrity} wins {award} for performance in {movie}",
    "{show} renewed for {ordinal} season",
    "{musician} announces world tour starting in {month}",
    "{movie} receives {number} nominations at {award} awards",
    "{celebrity} stars in new {genre} film {movie}",
    "{show} finale draws {number} million viewers",
    "{musician} drops surprise album featuring {number} tracks",
    "{celebrity} and {celebrity2} to star in {movie}",
    "Critics praise {movie} for groundbreaking {aspect}",
    "{show} streaming numbers break records on {platform}",
    "{musician} collaborates with {musician2} on new single",
    "{award} ceremony honors {celebrity} with lifetime achievement",
    "{movie} sequel announced with {celebrity} returning",
    "{show} creator discusses plans for final season",
    "Controversy surrounds {celebrity} comments at {award}",
    "{movie} franchise reaches {number} billion globally",
    "{musician} concert sells out in {number} minutes",
    "{show} spin-off series ordered by {platform}",
    "{celebrity} launches production company for {genre} projects",
    "Behind the scenes documentary reveals {movie} secrets",
    "{musician} breaks streaming record with {number} million plays",
    "{show} cast reunites for special anniversary episode",
    "{award} voters surprise with {movie} best picture win",
    "{celebrity} biography {movie} begins production",
    "Soundtrack from {movie} tops music charts",
    "{show} tackles controversial {aspect} in new episode",
    "{musician} postpones tour due to health concerns",
    "{celebrity} responds to criticism over {movie} role",
    "Virtual reality experience brings {movie} to life",
    "{show} introduces new characters in {ordinal} season",
    "{musician} releases deluxe edition with {number} bonus songs",
    "Film festival premiere for {celebrity} directorial debut",
    "{movie} remake updates classic with modern {aspect}",
    "{show} merchandise sales exceed {number} million",
    "{musician} charity concert raises funds for {aspect}",
    "{award} nominations announced with {movie} leading",
    "{celebrity} takes break from acting after {number} films",
    "Animated {movie} appeals to audiences of all ages",
    "{show} cliffhanger leaves fans demanding answers",
    "{musician} documentary explores rise to fame",
    "Casting announcement for {movie} sparks fan debate",
    "{celebrity} perfects {aspect} for demanding {movie} role",
    "{show} crosses {number} episode milestone",
    "{musician} tribute concert features surprise guests",
    "International distribution secured for {movie}",
    "{award} ceremony ratings climb to {number} million",
    "{celebrity} foundation supports aspiring {genre} artists",
    "{show} production moves to new {platform} home",
    "{movie} visual effects team pushes boundaries of {aspect}",
]

# EXPANDED VOCABULARIES (2-3x larger than basic)
LEADERS = [
    "President Smith", "Prime Minister Johnson", "Chancellor Weber", "President Martinez",
    "PM Anderson", "President Chen", "Prime Minister Patel", "Chancellor Rodriguez",
    "President Williams", "PM Thompson", "President Kim", "Prime Minister Davis",
    "Chancellor Brown", "President Wilson", "PM Garcia"
]

NAMES = [
    "Williams", "Chen", "Patel", "Rodriguez", "Kim", "Taylor", "Anderson", "Thomas",
    "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez",
    "Robinson", "Clark", "Lewis", "Lee", "Walker"
]

COUNTRIES = [
    "France", "Germany", "Japan", "Canada", "Brazil", "India", "Australia", "Mexico",
    "Italy", "Spain", "Korea", "Netherlands", "Sweden", "Norway", "Belgium",
    "Switzerland", "Austria", "Poland", "Turkey", "Argentina", "Chile", "Portugal"
]

PARTIES = [
    "Democratic", "Republican", "Labour", "Conservative", "Green", "Liberal",
    "Socialist", "Progressive", "Reform", "Unity", "Alliance", "Freedom"
]

POLICIES = [
    "healthcare", "education", "climate", "economic", "immigration", "tax", "trade",
    "energy", "defense", "infrastructure", "welfare", "housing", "transportation",
    "agriculture", "technology", "environmental", "justice"
]

TOPICS = [
    "healthcare", "climate change", "education", "economy", "security", "infrastructure",
    "employment", "energy", "trade", "immigration", "defense", "environment",
    "technology", "housing", "transportation", "agriculture", "justice"
]

AGREEMENTS = [
    "trade agreement", "peace treaty", "climate pact", "defense alliance",
    "cooperation deal", "security agreement", "economic partnership", "technology accord"
]

LOCATIONS = [
    "California", "Texas", "Ontario", "Bavaria", "Queensland", "New York",
    "Florida", "Victoria", "Catalonia", "Lombardy", "Provence", "Saxony"
]

TEAMS = [
    "Eagles", "Tigers", "Warriors", "United", "City", "Rangers", "Lions", "Sharks",
    "Bulls", "Jets", "Dragons", "Phoenix", "Thunder", "Storm", "Galaxy",
    "Titans", "Rockets", "Knights", "Hawks", "Wolves", "Panthers"
]

PLAYERS = [
    "Jackson", "Martinez", "Thompson", "Wilson", "Davis", "Brown", "Anderson", "Taylor",
    "Miller", "Garcia", "Rodriguez", "Lee", "Walker", "Hall", "Allen",
    "Young", "King", "Wright", "Lopez", "Hill", "Scott"
]

SPORTS = [
    "basketball", "football", "soccer", "baseball", "hockey", "tennis",
    "rugby", "cricket", "volleyball", "golf", "swimming"
]

TOURNAMENTS = [
    "World Cup", "Championship", "Super Bowl", "Finals", "Open",
    "Masters", "Series", "Classic", "Trophy", "Cup"
]

AWARDS = ["MVP", "rookie", "defensive", "offensive", "all-star"]

METRICS = ["assists", "yards", "goals", "saves", "rebounds", "points", "runs"]

ORDINALS = ["second", "third", "fourth", "fifth", "sixth", "seventh"]

RESULTS = ["devastating", "tough", "close", "unexpected", "disappointing", "difficult"]

COMPANIES = [
    "TechCorp", "DataSystems", "CloudNet", "AppWorks", "ByteLogic", "CyberDynamics",
    "NexGen", "InnovateLab", "QuantumTech", "SoftBridge", "DigiCore", "CodeForge",
    "NetSphere", "InfoStream", "SystemPro", "DataVault"
]

PRODUCTS = [
    "smartphone", "laptop", "tablet", "smartwatch", "software", "app", "platform",
    "device", "processor", "service", "console", "headset", "router", "server"
]

FEATURES = [
    "AI", "security", "speed", "battery", "camera", "voice control", "5G",
    "connectivity", "performance", "storage", "display", "biometric"
]

TECHNOLOGIES = [
    "artificial intelligence", "machine learning", "blockchain", "cloud computing",
    "robotics", "5G", "quantum computing", "virtual reality", "augmented reality",
    "internet of things", "edge computing", "neural networks"
]

APPLICATIONS = [
    "healthcare", "education", "manufacturing", "transportation", "agriculture",
    "finance", "retail", "logistics", "entertainment", "communication"
]

INDUSTRIES = [
    "healthcare", "finance", "retail", "automotive", "education", "manufacturing",
    "logistics", "entertainment", "telecommunications", "energy"
]

DEMOGRAPHICS = ["business", "student", "senior", "mobile", "enterprise", "consumer"]

CELEBRITIES = [
    "Emma Stone", "Chris Evans", "Jennifer Lawrence", "Ryan Gosling", "Zendaya",
    "Tom Holland", "Scarlett Johansson", "Michael B Jordan", "Florence Pugh",
    "Timothee Chalamet", "Margot Robbie", "Oscar Isaac"
]

MOVIES = [
    "Starlight", "The Journey", "Dark Waters", "Lost City", "New Dawn", "Echo",
    "Horizon", "Midnight Sun", "Rising Tide", "Silver Lining", "Eternal", "Cascade"
]

SHOWS = [
    "The Crown", "Westworld", "Stranger Things", "The Office", "Breaking Bad",
    "Succession", "The Wire", "Mad Men", "Ozark", "The Mandalorian"
]

MUSICIANS = [
    "Taylor Swift", "Drake", "Beyonce", "Ed Sheeran", "Billie Eilish",
    "The Weeknd", "Ariana Grande", "Post Malone", "Dua Lipa", "Bad Bunny"
]

AWARDS_ENT = ["Oscar", "Emmy", "Golden Globe", "Grammy", "SAG", "Tony", "BAFTA"]

GENRES = ["action", "drama", "comedy", "thriller", "sci-fi", "horror", "romance"]

PLATFORMS = ["Netflix", "Disney Plus", "HBO Max", "Amazon Prime", "Apple TV", "Hulu"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

ASPECTS = [
    "cinematography", "storytelling", "visual effects", "direction",
    "performances", "soundtrack", "production design", "writing"
]


def fill_template(template, category):
    """Fill template with random values"""
    filled = template

    # Track used values to avoid repetition within same headline
    used_teams = []
    used_countries = []
    used_companies = []
    used_celebrities = []
    used_leaders = []

    # Replace placeholders with random values
    replacements = {
        '{leader}': lambda: random.choice([l for l in LEADERS if l not in used_leaders] or LEADERS),
        '{leader2}': lambda: random.choice([l for l in LEADERS if l not in used_leaders] or LEADERS),
        '{name}': lambda: random.choice(NAMES),
        '{country}': lambda: random.choice([c for c in COUNTRIES if c not in used_countries] or COUNTRIES),
        '{country2}': lambda: random.choice([c for c in COUNTRIES if c not in used_countries] or COUNTRIES),
        '{party}': lambda: random.choice(PARTIES),
        '{policy}': lambda: random.choice(POLICIES),
        '{topic}': lambda: random.choice(TOPICS),
        '{agreement}': lambda: random.choice(AGREEMENTS),
        '{location}': lambda: random.choice(LOCATIONS),
        '{issue}': lambda: random.choice(TOPICS),
        '{team}': lambda: random.choice([t for t in TEAMS if t not in used_teams] or TEAMS),
        '{team2}': lambda: random.choice([t for t in TEAMS if t not in used_teams] or TEAMS),
        '{player}': lambda: random.choice(PLAYERS),
        '{sport}': lambda: random.choice(SPORTS),
        '{tournament}': lambda: random.choice(TOURNAMENTS),
        '{award}': lambda: random.choice(AWARDS),
        '{metric}': lambda: random.choice(METRICS),
        '{ordinal}': lambda: random.choice(ORDINALS),
        '{result}': lambda: random.choice(RESULTS),
        '{company}': lambda: random.choice([c for c in COMPANIES if c not in used_companies] or COMPANIES),
        '{company2}': lambda: random.choice([c for c in COMPANIES if c not in used_companies] or COMPANIES),
        '{product}': lambda: random.choice(PRODUCTS),
        '{feature}': lambda: random.choice(FEATURES),
        '{technology}': lambda: random.choice(TECHNOLOGIES),
        '{application}': lambda: random.choice(APPLICATIONS),
        '{industry}': lambda: random.choice(INDUSTRIES),
        '{demographic}': lambda: random.choice(DEMOGRAPHICS),
        '{celebrity}': lambda: random.choice([c for c in CELEBRITIES if c not in used_celebrities] or CELEBRITIES),
        '{celebrity2}': lambda: random.choice([c for c in CELEBRITIES if c not in used_celebrities] or CELEBRITIES),
        '{movie}': lambda: random.choice(MOVIES),
        '{show}': lambda: random.choice(SHOWS),
        '{musician}': lambda: random.choice(MUSICIANS),
        '{musician2}': lambda: random.choice(MUSICIANS),
        '{genre}': lambda: random.choice(GENRES),
        '{platform}': lambda: random.choice(PLATFORMS),
        '{month}': lambda: random.choice(MONTHS),
        '{aspect}': lambda: random.choice(ASPECTS),
        '{number}': lambda: str(random.choice([2, 3, 5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200, 250, 500])),
        '{number2}': lambda: str(random.choice([1, 2, 3, 5, 10, 15, 20, 25])),
    }

    # Apply replacements
    for placeholder, generator in replacements.items():
        if placeholder in filled:
            value = generator()
            filled = filled.replace(placeholder, value, 1)

            # Track used values
            if placeholder == '{team}' or placeholder == '{team2}':
                used_teams.append(value)
            elif placeholder == '{country}' or placeholder == '{country2}':
                used_countries.append(value)
            elif placeholder == '{company}' or placeholder == '{company2}':
                used_companies.append(value)
            elif placeholder == '{celebrity}' or placeholder == '{celebrity2}':
                used_celebrities.append(value)
            elif placeholder == '{leader}' or placeholder == '{leader2}':
                used_leaders.append(value)

    return filled


def generate_headlines(n_per_category=2500):
    """Generate extended headlines dataset"""

    categories = {
        'Politics': POLITICS_TEMPLATES,
        'Sports': SPORTS_TEMPLATES,
        'Technology': TECHNOLOGY_TEMPLATES,
        'Entertainment': ENTERTAINMENT_TEMPLATES
    }

    headlines = []
    headline_id = 1

    print(f"Generating {n_per_category * 4} headlines...")

    for category, templates in categories.items():
        print(f"  Generating {n_per_category} {category} headlines...")
        for i in range(n_per_category):
            template = random.choice(templates)
            headline = fill_template(template, category)

            # Count words
            words = headline.split()
            word_count = len(words)

            # Check if contains number
            has_number = bool(re.search(r'\d', headline))

            headlines.append({
                'headline_id': headline_id,
                'headline': headline,
                'category': category,
                'word_count': word_count,
                'has_number': has_number
            })

            headline_id += 1

            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{n_per_category} completed")

    return pd.DataFrame(headlines)


if __name__ == "__main__":
    # Generate dataset
    print("=" * 60)
    print("LARGE NEWS HEADLINES GENERATOR")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    df = generate_headlines(n_per_category=6250)

    # Shuffle the dataset
    print("\nShuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Update headline_id after shuffle
    df['headline_id'] = range(1, len(df) + 1)

    # Save to CSV
    output_file = '../large/news_headlines_large.csv'
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total headlines: {len(df)}")
    print(f"Total words: {df['word_count'].sum():,}")
    print(f"Average words per headline: {df['word_count'].mean():.2f}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    print(f"\nWord count statistics:")
    print(df['word_count'].describe())
    print(f"\nHeadlines with numbers: {df['has_number'].sum()} ({df['has_number'].mean()*100:.1f}%)")
    print(f"\nFirst 10 headlines:")
    print(df.head(10)[['headline_id', 'headline', 'category']])
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
