"""
Generate synthetic news headlines dataset for NLP homework
Creates 400 headlines across 4 categories
"""

import pandas as pd
import random
import re

# Set seed for reproducibility
random.seed(42)

# Templates for each category
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
]

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
]

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
]

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
]

# Data for filling templates
LEADERS = ["President Smith", "Prime Minister Johnson", "Chancellor Weber", "President Martinez", "PM Anderson"]
NAMES = ["Williams", "Chen", "Patel", "Rodriguez", "Kim"]
COUNTRIES = ["France", "Germany", "Japan", "Canada", "Brazil", "India", "Australia", "Mexico"]
PARTIES = ["Democratic", "Republican", "Labour", "Conservative", "Green"]
POLICIES = ["healthcare", "education", "climate", "economic", "immigration", "tax", "trade", "energy"]
TOPICS = ["healthcare", "climate change", "education", "economy", "security", "infrastructure", "employment"]
AGREEMENTS = ["trade agreement", "peace treaty", "climate pact", "defense alliance"]
LOCATIONS = ["California", "Texas", "Ontario", "Bavaria", "Queensland"]

TEAMS = ["Eagles", "Tigers", "Warriors", "United", "City", "Rangers", "Lions", "Sharks", "Bulls", "Jets"]
PLAYERS = ["Jackson", "Martinez", "Thompson", "Wilson", "Davis", "Brown", "Anderson", "Taylor"]
SPORTS = ["basketball", "football", "soccer", "baseball", "hockey", "tennis"]
TOURNAMENTS = ["World Cup", "Championship", "Super Bowl", "Finals", "Open"]
AWARDS = ["MVP", "rookie", "defensive", "offensive"]
METRICS = ["assists", "yards", "goals", "saves", "rebounds"]
ORDINALS = ["second", "third", "fourth", "fifth"]

COMPANIES = ["TechCorp", "DataSystems", "CloudNet", "AppWorks", "ByteLogic", "CyberDynamics"]
PRODUCTS = ["smartphone", "laptop", "tablet", "smartwatch", "software", "app", "platform", "device"]
FEATURES = ["AI", "security", "speed", "battery", "camera", "voice control", "5G"]
TECHNOLOGIES = ["artificial intelligence", "machine learning", "blockchain", "cloud computing", "robotics", "5G"]
APPLICATIONS = ["healthcare", "education", "manufacturing", "transportation", "agriculture"]
INDUSTRIES = ["healthcare", "finance", "retail", "automotive", "education"]
DEMOGRAPHICS = ["business", "student", "senior", "mobile"]

CELEBRITIES = ["Emma Stone", "Chris Evans", "Jennifer Lawrence", "Ryan Gosling", "Zendaya", "Tom Holland"]
MOVIES = ["Starlight", "The Journey", "Dark Waters", "Lost City", "New Dawn", "Echo"]
SHOWS = ["The Crown", "Westworld", "Stranger Things", "The Office", "Breaking Bad"]
MUSICIANS = ["Taylor Swift", "Drake", "Beyonce", "Ed Sheeran", "Billie Eilish"]
AWARDS = ["Oscar", "Emmy", "Golden Globe", "Grammy", "SAG"]
GENRES = ["action", "drama", "comedy", "thriller", "sci-fi"]
PLATFORMS = ["Netflix", "Disney Plus", "HBO Max", "Amazon Prime"]
MONTHS = ["January", "March", "June", "September", "November"]
ASPECTS = ["cinematography", "storytelling", "visual effects", "direction"]

def fill_template(template, category):
    """Fill template with random values"""
    filled = template

    # Replace placeholders with random values
    replacements = {
        '{leader}': lambda: random.choice(LEADERS),
        '{leader2}': lambda: random.choice([l for l in LEADERS if l != filled.split()[0] if '{leader}' in template]),
        '{name}': lambda: random.choice(NAMES),
        '{country}': lambda: random.choice(COUNTRIES),
        '{country2}': lambda: random.choice([c for c in COUNTRIES]),
        '{party}': lambda: random.choice(PARTIES),
        '{policy}': lambda: random.choice(POLICIES),
        '{topic}': lambda: random.choice(TOPICS),
        '{agreement}': lambda: random.choice(AGREEMENTS),
        '{location}': lambda: random.choice(LOCATIONS),
        '{issue}': lambda: random.choice(TOPICS),
        '{team}': lambda: random.choice(TEAMS),
        '{team2}': lambda: random.choice([t for t in TEAMS]),
        '{player}': lambda: random.choice(PLAYERS),
        '{sport}': lambda: random.choice(SPORTS),
        '{tournament}': lambda: random.choice(TOURNAMENTS),
        '{award}': lambda: random.choice(AWARDS),
        '{metric}': lambda: random.choice(METRICS),
        '{ordinal}': lambda: random.choice(ORDINALS),
        '{result}': lambda: random.choice(['devastating', 'tough', 'close', 'unexpected']),
        '{company}': lambda: random.choice(COMPANIES),
        '{company2}': lambda: random.choice([c for c in COMPANIES]),
        '{product}': lambda: random.choice(PRODUCTS),
        '{feature}': lambda: random.choice(FEATURES),
        '{technology}': lambda: random.choice(TECHNOLOGIES),
        '{application}': lambda: random.choice(APPLICATIONS),
        '{industry}': lambda: random.choice(INDUSTRIES),
        '{demographic}': lambda: random.choice(DEMOGRAPHICS),
        '{celebrity}': lambda: random.choice(CELEBRITIES),
        '{celebrity2}': lambda: random.choice([c for c in CELEBRITIES]),
        '{movie}': lambda: random.choice(MOVIES),
        '{show}': lambda: random.choice(SHOWS),
        '{musician}': lambda: random.choice(MUSICIANS),
        '{musician2}': lambda: random.choice([m for m in MUSICIANS]),
        '{genre}': lambda: random.choice(GENRES),
        '{platform}': lambda: random.choice(PLATFORMS),
        '{month}': lambda: random.choice(MONTHS),
        '{aspect}': lambda: random.choice(ASPECTS),
        '{number}': lambda: str(random.choice([2, 3, 5, 10, 15, 20, 25, 30, 50, 100, 150])),
        '{number2}': lambda: str(random.choice([1, 2, 3, 5, 10, 15, 20])),
    }

    # Apply replacements
    for placeholder, generator in replacements.items():
        if placeholder in filled:
            filled = filled.replace(placeholder, generator(), 1)

    return filled

def generate_headlines(n_per_category=100):
    """Generate synthetic headlines dataset"""

    categories = {
        'Politics': POLITICS_TEMPLATES,
        'Sports': SPORTS_TEMPLATES,
        'Technology': TECHNOLOGY_TEMPLATES,
        'Entertainment': ENTERTAINMENT_TEMPLATES
    }

    headlines = []
    headline_id = 1

    for category, templates in categories.items():
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

    return pd.DataFrame(headlines)

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic news headlines dataset...")
    df = generate_headlines(n_per_category=100)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Update headline_id after shuffle
    df['headline_id'] = range(1, len(df) + 1)

    # Save to CSV
    output_file = 'news_headlines_dataset.csv'
    df.to_csv(output_file, index=False)

    print(f"Dataset created successfully: {output_file}")
    print(f"Total headlines: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    print(f"\nFirst few headlines:")
    print(df.head(10))
