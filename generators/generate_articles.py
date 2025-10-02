"""
Generate news articles dataset with matching headlines for summarization tasks
Creates 1,000 article-headline pairs across 4 categories
"""

import pandas as pd
import random
import re
from datetime import datetime

# Set seed for reproducibility
random.seed(42)

# Discourse connectives for coherence
CONNECTIVES = {
    'addition': ['Additionally', 'Furthermore', 'Moreover', 'In addition'],
    'contrast': ['However', 'Nevertheless', 'Meanwhile', 'On the other hand'],
    'cause': ['Therefore', 'Consequently', 'As a result', 'Thus'],
    'temporal': ['Earlier', 'Previously', 'Later', 'Subsequently'],
    'elaboration': ['In fact', 'Specifically', 'Indeed', 'Notably'],
}

# Pronouns for entity reference
PRONOUNS = {
    'person': ['he', 'she', 'they'],
    'org': ['the company', 'the organization', 'the team', 'the group'],
    'country': ['the nation', 'the country'],
}

# POLITICS ARTICLE TEMPLATES
POLITICS_ARTICLE_TEMPLATES = [
    {
        'headline': "{leader} announces major {policy} reform initiative",
        'sentences': [
            "{leader} announced a comprehensive {policy} reform initiative during a press conference on {day}.",
            "The proposal includes a {number} billion investment over the next {number2} years to address key challenges in the sector.",
            "{discourse} officials from the {party} party have expressed strong support for the measure.",
            "Critics argue that the timeline for implementation is too ambitious, citing concerns about funding and logistics.",
            "The initiative will be debated in parliament next month, with a vote expected before the end of the quarter."
        ]
    },
    {
        'headline': "{country} and {country2} reach historic trade agreement",
        'sentences': [
            "{country} and {country2} signed a landmark trade agreement aimed at boosting economic cooperation between the two nations.",
            "The deal eliminates tariffs on over {number} categories of goods and establishes new frameworks for digital commerce.",
            "{discourse} trade representatives emphasized that the agreement will create thousands of jobs in both countries.",
            "Industry leaders have welcomed the news, particularly in the manufacturing and technology sectors.",
            "The agreement is expected to take effect within {number2} months pending legislative approval."
        ]
    },
    {
        'headline': "Parliament debates controversial {topic} legislation",
        'sentences': [
            "Heated debates erupted in parliament today as lawmakers discussed sweeping {topic} legislation.",
            "The proposed bill would significantly alter current regulations and has divided opinion along party lines.",
            "{discourse} the {party} party argues the changes are necessary to address emerging challenges.",
            "Opposition members have vowed to block the measure, calling it hasty and potentially harmful.",
            "A final vote is scheduled for next week, with the outcome remaining uncertain."
        ]
    },
    {
        'headline': "{party} party secures majority in {location} election",
        'sentences': [
            "The {party} party won a decisive victory in {location} elections, securing a clear majority.",
            "Voter turnout reached {number} percent, one of the highest levels in recent history.",
            "{discourse} party leaders credited their success to focusing on {topic} and {issue} issues.",
            "The defeated {party2} party conceded defeat and pledged to work constructively in opposition.",
            "The new government is expected to be sworn in within the next {number2} weeks."
        ]
    },
    {
        'headline': "Government unveils {number} billion {topic} spending plan",
        'sentences': [
            "The government released details of a {number} billion spending package focused on {topic} priorities.",
            "Key allocations include funding for infrastructure improvements, research initiatives, and public services.",
            "{discourse} the finance minister defended the plan as fiscally responsible despite the large price tag.",
            "Economic analysts are divided on whether the spending will achieve its stated goals.",
            "The proposal requires parliamentary approval, with discussions set to begin next month."
        ]
    },
]

# SPORTS ARTICLE TEMPLATES
SPORTS_ARTICLE_TEMPLATES = [
    {
        'headline': "{team} defeats {team2} in thrilling {tournament} final",
        'sentences': [
            "{team} emerged victorious over {team2} in a dramatic {tournament} final that went down to the wire.",
            "Star player {player} delivered a clutch performance, contributing {number} points in the closing minutes.",
            "{discourse} {team2} fought back valiantly but ultimately fell short despite their best efforts.",
            "The victory marks {team}'s {ordinal} {tournament} title and cements their status as a dominant force.",
            "Celebrations erupted among fans as the team lifted the trophy at the packed stadium."
        ]
    },
    {
        'headline': "{player} breaks {sport} record with outstanding performance",
        'sentences': [
            "{player} made history today by breaking the long-standing {sport} record for {metric}.",
            "The athlete's performance of {number} {metric} surpassed the previous mark set {number2} years ago.",
            "{discourse} teammates and coaches praised {player}'s dedication and exceptional talent.",
            "The record-breaking moment came in the {ordinal} period, sending the crowd into a frenzy.",
            "{player} expressed gratitude to supporters and vowed to continue pushing boundaries in the sport."
        ]
    },
    {
        'headline': "{team} secures playoff berth with crucial victory",
        'sentences': [
            "{team} clinched a playoff spot with a hard-fought win over {team2} in a must-win game.",
            "The {number} to {number2} victory came after weeks of intense competition and nail-biting finishes.",
            "{discourse} coach {name} commended the team's resilience and determination throughout the season.",
            "Key contributions from {player} and strong defensive play proved decisive in the outcome.",
            "The team will now prepare for the playoffs, where they will face tougher challenges ahead."
        ]
    },
    {
        'headline': "{player} announces retirement after illustrious career",
        'sentences': [
            "Legendary {sport} player {player} announced retirement today, ending a career spanning {number} years.",
            "The athlete leaves behind an impressive legacy, including multiple {tournament} championships and individual awards.",
            "{discourse} fans and fellow players paid tribute to {player}'s contributions to the sport.",
            "In an emotional press conference, {player} thanked teammates, coaches, and supporters.",
            "The sports world will remember {player} as one of the greatest to ever play the game."
        ]
    },
    {
        'headline': "Underdog {team} stuns favorites {team2} in upset win",
        'sentences': [
            "In one of the biggest upsets of the season, underdog {team} defeated heavily favored {team2}.",
            "Few experts gave {team} a chance, but they defied expectations with a dominant performance.",
            "{discourse} {team2} struggled throughout the game, unable to find their usual rhythm.",
            "The victory sends shockwaves through the league and changes the playoff picture dramatically.",
            "{team} players celebrated the historic win as fans stormed the field in jubilation."
        ]
    },
]

# TECHNOLOGY ARTICLE TEMPLATES
TECHNOLOGY_ARTICLE_TEMPLATES = [
    {
        'headline': "{company} launches revolutionary {product} with {feature} technology",
        'sentences': [
            "{company} unveiled its latest {product} featuring groundbreaking {feature} technology at a major event today.",
            "The device promises to deliver {number} percent better performance compared to previous generation models.",
            "{discourse} industry analysts predict the product will significantly impact the {industry} market.",
            "Pre-orders have already exceeded expectations, with initial stock selling out within {number2} hours.",
            "The {product} will officially go on sale next month, with availability expanding to {number} countries."
        ]
    },
    {
        'headline': "{company} and {company2} announce strategic partnership",
        'sentences': [
            "{company} and {company2} have formed a strategic partnership to advance {technology} development.",
            "The collaboration will combine {company}'s expertise in {application} with {company2}'s innovations in {feature}.",
            "{discourse} both companies see tremendous potential in working together to accelerate progress.",
            "The partnership includes a joint investment of {number} million in research and development.",
            "First products resulting from the collaboration are expected to launch within {number2} years."
        ]
    },
    {
        'headline': "Study reveals {technology} adoption growing rapidly",
        'sentences': [
            "A new research study shows {technology} adoption has increased by {number} percent over the past year.",
            "The findings indicate growing acceptance among {demographic} users, particularly in {application} applications.",
            "{discourse} researchers attribute the growth to improved accessibility and declining costs.",
            "Major players in the {industry} sector are investing heavily to capitalize on the trend.",
            "Experts predict the {technology} market will reach {number} billion in value within {number2} years."
        ]
    },
    {
        'headline': "{company} addresses security concerns in popular {product}",
        'sentences': [
            "{company} released an urgent security update for its {product} after vulnerabilities were discovered.",
            "The issues affected an estimated {number} million users worldwide and prompted immediate action.",
            "{discourse} cybersecurity experts had alerted the company to potential exploits several weeks ago.",
            "The company apologized to customers and assured them that no data breaches have been detected.",
            "All users are strongly advised to install the latest update to protect their devices and information."
        ]
    },
    {
        'headline': "AI-powered {product} transforms {industry} operations",
        'sentences': [
            "A new artificial intelligence powered {product} is revolutionizing how companies in the {industry} sector operate.",
            "Early adopters report efficiency gains of up to {number} percent and significant cost reductions.",
            "{discourse} the technology uses advanced {feature} capabilities to automate complex tasks.",
            "Industry leaders believe AI integration represents the future of {industry} and competitive advantage.",
            "Market analysis suggests widespread adoption could occur within the next {number2} years."
        ]
    },
]

# ENTERTAINMENT ARTICLE TEMPLATES
ENTERTAINMENT_ARTICLE_TEMPLATES = [
    {
        'headline': "{movie} dominates box office with {number} million opening",
        'sentences': [
            "The highly anticipated {genre} film {movie} shattered box office records with a {number} million opening weekend.",
            "Starring {celebrity} and {celebrity2}, the movie exceeded industry projections and delighted audiences.",
            "{discourse} critics have praised the film's innovative {aspect} and compelling storytelling.",
            "The success represents a major win for the studio, which invested heavily in production and marketing.",
            "Industry insiders predict {movie} could become one of the highest-grossing films of the year."
        ]
    },
    {
        'headline': "{celebrity} wins {award} for outstanding performance in {movie}",
        'sentences': [
            "{celebrity} took home the prestigious {award} award for a powerful performance in {movie}.",
            "The emotional acceptance speech drew a standing ovation from fellow nominees and industry peers.",
            "{discourse} this marks {celebrity}'s {ordinal} {award} win, cementing a reputation as one of the finest actors.",
            "The role required months of preparation and dedication, including extensive training and research.",
            "Fans and critics alike celebrated the well-deserved recognition for {celebrity}'s exceptional work."
        ]
    },
    {
        'headline': "{show} renewed for {ordinal} season by {platform}",
        'sentences': [
            "Streaming giant {platform} announced it has renewed the hit series {show} for a {ordinal} season.",
            "The show has consistently topped viewership charts since its premiere, attracting millions of subscribers.",
            "{discourse} creators expressed excitement about continuing to explore the show's rich narrative universe.",
            "Production on the new season is set to begin next month, with a release planned for next year.",
            "Cast members shared their enthusiasm on social media, thanking fans for their unwavering support."
        ]
    },
    {
        'headline': "{musician} announces world tour with {number} concert dates",
        'sentences': [
            "International superstar {musician} revealed plans for a massive world tour featuring {number} shows across {number2} countries.",
            "The tour will kick off in {month} and continue through the end of the year.",
            "{discourse} tickets went on sale this morning and are selling rapidly, with several venues already sold out.",
            "Fans can expect to hear hits from recent albums as well as classic favorites.",
            "Special guest performers will be announced in the coming weeks, adding to the excitement."
        ]
    },
    {
        'headline': "{award} ceremony celebrates excellence in {genre} entertainment",
        'sentences': [
            "The annual {award} awards ceremony honored the best in {genre} entertainment at a star-studded gala.",
            "Winners included {movie}, which took home {number} awards including best picture.",
            "{discourse} the ceremony featured memorable performances and heartfelt tributes to industry legends.",
            "Viewership for the televised event reached {number} million, up from last year.",
            "Next year's {award} awards are already generating buzz with several anticipated contenders."
        ]
    },
]

# Data pools for article generation
LEADERS = ["President Smith", "Prime Minister Johnson", "Chancellor Weber", "President Martinez", "PM Anderson"]
NAMES = ["Williams", "Chen", "Patel", "Rodriguez", "Kim", "Taylor", "Anderson", "Thomas"]
COUNTRIES = ["France", "Germany", "Japan", "Canada", "Brazil", "India", "Australia", "Mexico", "Italy", "Spain"]
PARTIES = ["Democratic", "Republican", "Labour", "Conservative", "Green", "Liberal", "Progressive"]
PARTY2_POOL = ["Reform", "Unity", "Alliance", "Freedom", "Socialist"]
POLICIES = ["healthcare", "education", "climate", "economic", "immigration", "tax", "trade", "energy"]
TOPICS = ["healthcare", "climate change", "education", "economy", "security", "infrastructure", "employment"]
ISSUES = ["housing", "transportation", "technology", "environment", "defense"]
LOCATIONS = ["California", "Texas", "Ontario", "Bavaria", "Queensland", "New York", "Florida"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TEAMS = ["Eagles", "Tigers", "Warriors", "United", "City", "Rangers", "Lions", "Sharks", "Bulls", "Jets", "Dragons", "Phoenix"]
PLAYERS = ["Jackson", "Martinez", "Thompson", "Wilson", "Davis", "Brown", "Anderson", "Taylor", "Miller", "Garcia"]
SPORTS = ["basketball", "football", "soccer", "baseball", "hockey", "tennis"]
TOURNAMENTS = ["Championship", "Cup", "Finals", "Series", "Trophy", "Classic"]
METRICS = ["points", "assists", "yards", "goals", "saves", "rebounds"]
ORDINALS = ["second", "third", "fourth", "fifth", "sixth"]
COMPANIES = ["TechCorp", "DataSystems", "CloudNet", "AppWorks", "ByteLogic", "CyberDynamics", "NexGen", "InnovateLab"]
PRODUCTS = ["smartphone", "laptop", "tablet", "smartwatch", "software", "app", "platform", "device", "service"]
FEATURES = ["AI", "security", "speed", "battery", "camera", "voice control", "5G", "connectivity", "biometric"]
TECHNOLOGIES = ["artificial intelligence", "machine learning", "blockchain", "cloud computing", "robotics", "quantum computing"]
APPLICATIONS = ["healthcare", "education", "manufacturing", "transportation", "finance", "retail", "logistics"]
INDUSTRIES = ["healthcare", "finance", "retail", "automotive", "education", "manufacturing", "telecommunications"]
DEMOGRAPHICS = ["business", "enterprise", "consumer", "student", "mobile"]
CELEBRITIES = ["Emma Stone", "Chris Evans", "Jennifer Lawrence", "Ryan Gosling", "Zendaya", "Tom Holland", "Margot Robbie"]
MOVIES = ["Starlight", "The Journey", "Dark Waters", "Lost City", "New Dawn", "Echo", "Horizon", "Midnight Sun"]
SHOWS = ["The Crown", "Westworld", "Stranger Things", "The Office", "Succession", "The Wire", "Ozark"]
MUSICIANS = ["Taylor Swift", "Drake", "Beyonce", "Ed Sheeran", "Billie Eilish", "The Weeknd", "Ariana Grande"]
AWARDS = ["Oscar", "Emmy", "Golden Globe", "Grammy", "SAG", "Tony", "BAFTA"]
GENRES = ["action", "drama", "comedy", "thriller", "sci-fi", "horror", "romance"]
PLATFORMS = ["Netflix", "Disney Plus", "HBO Max", "Amazon Prime", "Apple TV", "Hulu"]
MONTHS = ["January", "March", "May", "June", "September", "October", "November"]
ASPECTS = ["cinematography", "storytelling", "visual effects", "direction", "performances", "soundtrack"]


def fill_article_template(template, category):
    """Fill article template with random values and ensure coherence"""

    # Select discourse connective
    discourse_type = random.choice(list(CONNECTIVES.keys()))
    discourse = random.choice(CONNECTIVES[discourse_type])

    # Create replacement dictionary with consistent values
    replacements = {
        '{leader}': random.choice(LEADERS),
        '{leader2}': random.choice(LEADERS),
        '{name}': random.choice(NAMES),
        '{country}': random.choice(COUNTRIES),
        '{country2}': random.choice(COUNTRIES),
        '{party}': random.choice(PARTIES),
        '{party2}': random.choice(PARTY2_POOL),
        '{policy}': random.choice(POLICIES),
        '{topic}': random.choice(TOPICS),
        '{issue}': random.choice(ISSUES),
        '{location}': random.choice(LOCATIONS),
        '{day}': random.choice(DAYS),
        '{team}': random.choice(TEAMS),
        '{team2}': random.choice(TEAMS),
        '{player}': random.choice(PLAYERS),
        '{sport}': random.choice(SPORTS),
        '{tournament}': random.choice(TOURNAMENTS),
        '{metric}': random.choice(METRICS),
        '{ordinal}': random.choice(ORDINALS),
        '{company}': random.choice(COMPANIES),
        '{company2}': random.choice(COMPANIES),
        '{product}': random.choice(PRODUCTS),
        '{feature}': random.choice(FEATURES),
        '{technology}': random.choice(TECHNOLOGIES),
        '{application}': random.choice(APPLICATIONS),
        '{industry}': random.choice(INDUSTRIES),
        '{demographic}': random.choice(DEMOGRAPHICS),
        '{celebrity}': random.choice(CELEBRITIES),
        '{celebrity2}': random.choice(CELEBRITIES),
        '{movie}': random.choice(MOVIES),
        '{show}': random.choice(SHOWS),
        '{musician}': random.choice(MUSICIANS),
        '{award}': random.choice(AWARDS),
        '{genre}': random.choice(GENRES),
        '{platform}': random.choice(PLATFORMS),
        '{month}': random.choice(MONTHS),
        '{aspect}': random.choice(ASPECTS),
        '{number}': str(random.choice([5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200, 250])),
        '{number2}': str(random.choice([2, 3, 4, 5, 10, 12, 15, 18, 24])),
        '{discourse}': discourse,
    }

    # Fill headline
    headline = template['headline']
    for key, value in replacements.items():
        headline = headline.replace(key, value)

    # Fill sentences
    article_sentences = []
    for sentence in template['sentences']:
        filled_sentence = sentence
        for key, value in replacements.items():
            filled_sentence = filled_sentence.replace(key, value)
        article_sentences.append(filled_sentence)

    # Combine into article
    article = ' '.join(article_sentences)

    return headline, article


def generate_articles(n_per_category=250):
    """Generate articles dataset with headlines"""

    categories = {
        'Politics': POLITICS_ARTICLE_TEMPLATES,
        'Sports': SPORTS_ARTICLE_TEMPLATES,
        'Technology': TECHNOLOGY_ARTICLE_TEMPLATES,
        'Entertainment': ENTERTAINMENT_ARTICLE_TEMPLATES
    }

    articles = []
    article_id = 1

    print(f"Generating {n_per_category * 4} article-headline pairs...")

    for category, templates in categories.items():
        print(f"  Generating {n_per_category} {category} articles...")
        for i in range(n_per_category):
            template = random.choice(templates)
            headline, article = fill_article_template(template, category)

            # Count words
            article_words = article.split()
            headline_words = headline.split()
            article_word_count = len(article_words)
            headline_word_count = len(headline_words)

            # Check for numbers
            has_number = bool(re.search(r'\d', article))

            articles.append({
                'article_id': article_id,
                'headline': headline,
                'article': article,
                'category': category,
                'headline_word_count': headline_word_count,
                'article_word_count': article_word_count,
                'has_number': has_number
            })

            article_id += 1

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{n_per_category} completed")

    return pd.DataFrame(articles)


if __name__ == "__main__":
    # Generate dataset
    print("=" * 60)
    print("NEWS ARTICLES GENERATOR")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    df = generate_articles(n_per_category=250)

    # Shuffle the dataset
    print("\nShuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Update article_id after shuffle
    df['article_id'] = range(1, len(df) + 1)

    # Save to CSV
    output_file = '../articles/news_articles_dataset.csv'
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total articles: {len(df)}")
    print(f"Total words in articles: {df['article_word_count'].sum():,}")
    print(f"Total words in headlines: {df['headline_word_count'].sum():,}")
    print(f"Average article length: {df['article_word_count'].mean():.2f} words")
    print(f"Average headline length: {df['headline_word_count'].mean():.2f} words")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts().sort_index())
    print(f"\nArticle word count statistics:")
    print(df['article_word_count'].describe())
    print(f"\nArticles with numbers: {df['has_number'].sum()} ({df['has_number'].mean()*100:.1f}%)")
    print(f"\nFirst 3 article-headline pairs:")
    for idx in range(3):
        print(f"\n{idx+1}. Headline: {df.iloc[idx]['headline']}")
        print(f"   Article: {df.iloc[idx]['article'][:150]}...")
        print(f"   Category: {df.iloc[idx]['category']}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
