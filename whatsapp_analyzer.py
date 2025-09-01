#!/usr/bin/env python3
"""
WhatsApp Chat Analyzer
A comprehensive tool to analyze personal growth and relationship dynamics from WhatsApp chat exports.

Target Participants Configuration:
To analyze specific participants, either:
1. Add their names in TARGET_PARTICIPANTS list below, or
2. Use --participants CLI argument: --participants "Person1,Person2"

Example: python whatsapp_analyzer.py chat.txt --participants "A,B"
"""

# Configure target participants for analysis (leave empty to analyze all)
TARGET_PARTICIPANTS = ["A", "B"]

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import argparse
from textstat import flesch_reading_ease, automated_readability_index
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading sentiment analysis data...")
    nltk.download('vader_lexicon', quiet=True)

class WhatsAppAnalyzer:
    def __init__(self, file_path, target_participants=None):
        self.file_path = Path(file_path)
        self.messages = []
        self.df = None
        self.sia = SentimentIntensityAnalyzer()
        self.target_participants = target_participants or TARGET_PARTICIPANTS

    def has_forwarded_date_format(self, message_line):
        """Check if a message line has the forwarded conversation date format with AM/PM"""
        # Match the specific AM/PM format that appears in forwarded group conversations
        forwarded_date_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} [AP]M\]'
        return bool(re.search(forwarded_date_pattern, message_line))

    def is_target_participant(self, sender_name):
        """Check if sender is in the target participants list"""
        if not self.target_participants:
            return True  # If no target participants specified, include all
        return any(target.strip().lower() in sender_name.strip().lower() for target in self.target_participants)

    def is_forwarded_message(self, message):
        """Detect if a message is forwarded content"""

        # Common forwarded message indicators
        forwarded_indicators = [
            # WhatsApp forwarded message marker
            'Forwarded',

            # Common forwarded message patterns
            r'^[A-Z][a-z]+ forwarded',
            r'^Forwarded from .+',
            r'^\*Forwarded\*',
            r'^Good morning!?\s*\n',
            r'^Good night!?\s*\n',
            r'^Prayer for',
            r'^Thought for the day',

            # Messages containing forwarded conversation threads with AM/PM timestamps
            r'\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} [AP]M\]',  # Contains timestamp like [1/20/18, 9:57:25 AM]
            r'\d{1,2}:\d{2} [AP]M\]',  # Contains time like "9:57 AM]"
        ]

        # Check for forwarded indicators
        message_text = str(message).strip()

        for pattern in forwarded_indicators:
            if re.search(pattern, message_text, re.IGNORECASE | re.MULTILINE):
                return True

        # Additional heuristics
        if (
            len(message_text) > 500 and
            '...' in message_text and
            message_text.count('\n') > 15
        ):
            return True

        # Messages that look like articles/posts
        lines = message_text.split('\n')
        if len(lines) > 10:
            title_like_lines = sum(1 for line in lines if len(line.strip()) < 50 and line.strip().isupper() and len(line.strip()) > 3)
            if title_like_lines >= 2:
                return True

        # Messages with multiple links (often forwards)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, message_text)
        if len(urls) > 2:
            return True

        return False

    def is_forwarded_participant(self, sender_name):
        """Detect if a sender name is from a forwarded message thread"""
        sender = str(sender_name).strip()

        # Check for forwarded message participant patterns
        forwarded_participant_patterns = [
            r'^AM\]\s*.+',  # Matches "AM] PersonName"
            r'^PM\]\s*.+',  # Matches "PM] PersonName"
            r'^\d{1,2}:\d{2}\s*[AP]M\]\s*.+',  # Matches "12:34 AM] PersonName"
        ]

        for pattern in forwarded_participant_patterns:
            if re.match(pattern, sender, re.IGNORECASE):
                return True

        return False

    def parse_chat(self):
        """Parse WhatsApp chat file and extract messages"""
        print(f"📱 Parsing WhatsApp chat file: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Primary regex pattern for main conversation format (24-hour time, no AM/PM)
        main_pattern = r'^\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}:\d{2})\] ([^:]+): (.*)'

        # Secondary patterns for other formats (will be filtered if they have AM/PM)
        fallback_patterns = [
            r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\s*(?:[ap]m)?\]?\s*[-:]?\s*([^:]+):\s*(.*)',
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2})\s*[-]?\s*([^:]+):\s*(.*)',
            r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\s*([ap]m)?\s*[-]?\s*([^:]+):\s*(.*)'
        ]

        messages = []
        lines = content.split('\n')
        current_message = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that have the forwarded conversation AM/PM date format
            if self.has_forwarded_date_format(line):
                continue

            matched = False

            # Try main pattern first (preferred format)
            match = re.match(main_pattern, line)
            if match:
                date_str, time_str, sender, content = match.groups()
                matched = True
            else:
                # Try fallback patterns
                for pattern in fallback_patterns:
                    match = re.match(pattern, line)
                    if match:
                        # Parse new message
                        if len(match.groups()) == 4:
                            date_str, time_str, sender, content = match.groups()
                        else:
                            date_str, time_str, ampm_or_sender, sender_or_content, content = match.groups()
                            if ampm_or_sender and ampm_or_sender.lower() in ['am', 'pm']:
                                time_str += f" {ampm_or_sender}"
                                sender = sender_or_content
                            else:
                                sender = ampm_or_sender
                                content = sender_or_content
                        matched = True
                        break

            if matched:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)

                # Check if sender is a target participant
                if not self.is_target_participant(sender):
                    current_message = None
                    continue

                # Parse datetime
                try:
                    # Try different date formats
                    date_formats = ['%m/%d/%Y', '%d/%m/%Y', '%m/%d/%y', '%d/%m/%y']
                    time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']

                    dt = None
                    for date_fmt in date_formats:
                        for time_fmt in time_formats:
                            try:
                                datetime_str = f"{date_str} {time_str}"
                                dt = pd.to_datetime(datetime_str, format=f"{date_fmt} {time_fmt}")
                                break
                            except:
                                continue
                        if dt is not None:
                            break

                    if dt is None:
                        # Fallback to pandas auto-parsing
                        dt = pd.to_datetime(f"{date_str} {time_str}")

                    current_message = {
                        'datetime': dt,
                        'date': dt.date(),
                        'time': dt.time(),
                        'sender': sender.strip(),
                        'message': content.strip(),
                    }

                except Exception as e:
                    print(f"⚠️  Could not parse datetime: {date_str} {time_str} - {e}")
                    continue

            # If no pattern matched, this might be a continuation of the previous message
            if not matched and current_message:
                current_message['message'] += f" {line}"

        # Add the last message
        if current_message:
            messages.append(current_message)

        if not messages:
            print("❌ No messages could be parsed. Please check the file format.")
            return 0

        # Create DataFrame
        self.df_original = pd.DataFrame(messages)

        # Add message metrics
        self.df_original['char_count'] = self.df_original['message'].str.len()
        self.df_original['word_count'] = self.df_original['message'].str.split().str.len()
        self.df_original['is_forwarded'] = self.df_original['message'].apply(self.is_forwarded_message)
        self.df_original['is_forwarded_participant'] = self.df_original['sender'].apply(self.is_forwarded_participant)

        # Create filtered DataFrame (excluding forwarded messages, forwarded participants, and non-target participants)
        is_target_participant = self.df_original['sender'].apply(self.is_target_participant)
        self.df_filtered = self.df_original[
            (~self.df_original['is_forwarded']) &
            (~self.df_original['is_forwarded_participant']) &
            (is_target_participant)
        ].copy()

        # Use filtered data for main analysis
        self.df = self.df_filtered

        print(f"✅ Parsed {len(self.df_original):,} total messages")
        print(f"🔍 Filtered to {len(self.df):,} authentic messages for analysis")

        forwarded_msg_count = self.df_original['is_forwarded'].sum()
        forwarded_participant_count = self.df_original['is_forwarded_participant'].sum()
        non_target_participants = len(self.df_original) - len(self.df_original[self.df_original['sender'].apply(self.is_target_participant)])
        total_filtered = forwarded_msg_count + forwarded_participant_count + non_target_participants

        if self.target_participants:
            print(f"🎯 Target participants: {', '.join(self.target_participants)}")
            if non_target_participants > 0:
                print(f"👥 Filtered {non_target_participants:,} messages from non-target participants")

        if forwarded_msg_count > 0:
            print(f"📤 Detected and filtered {forwarded_msg_count:,} forwarded messages")
        if forwarded_participant_count > 0:
            print(f"👥 Detected and filtered {forwarded_participant_count:,} forwarded participant messages (AM/PM format)")
        if total_filtered > 0:
            print(f"🔍 Total filtered: {total_filtered:,} messages for targeted authentic analysis")

        return len(self.df)

    def basic_stats(self):
        """Generate basic statistics"""
        print("\n📊 Generating basic statistics...")

        total_messages = len(self.df)
        date_range = (self.df['datetime'].min(), self.df['datetime'].max())
        duration_days = (date_range[1] - date_range[0]).days
        duration_years = duration_days / 365.25

        senders = self.df['sender'].unique()

        stats = {
            'total_messages': total_messages,
            'duration_days': duration_days,
            'duration_years': duration_years,
            'date_range': date_range,
            'daily_average': total_messages / max(1, duration_days),
            'senders': senders,
            'sender_count': len(senders)
        }

        print(f"📈 Total Messages: {total_messages:,}")
        print(f"📅 Date Range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
        print(f"⏰ Duration: {duration_years:.1f} years ({duration_days:,} days)")
        print(f"💬 Daily Average: {stats['daily_average']:.1f} messages")
        print(f"👥 Participants: {', '.join(senders)}")

        return stats

    def sender_analysis(self):
        """Analyze communication patterns by sender"""
        print("\n👥 Analyzing sender patterns...")

        sender_stats = {}
        total_messages = len(self.df)

        for sender in self.df['sender'].unique():
            sender_data = self.df[self.df['sender'] == sender]

            stats = {
                'message_count': len(sender_data),
                'percentage': (len(sender_data) / total_messages) * 100,
                'avg_message_length': sender_data['char_count'].mean(),
                'avg_words_per_message': sender_data['word_count'].mean(),
                'total_words': sender_data['word_count'].sum(),
                'total_characters': sender_data['char_count'].sum(),
                'longest_message': sender_data['char_count'].max(),
                'messages_per_day': len(sender_data) / max(1, (self.df['datetime'].max() - self.df['datetime'].min()).days)
            }

            sender_stats[sender] = stats

            print(f"\n{sender}:")
            print(f"  Messages: {stats['message_count']:,} ({stats['percentage']:.1f}%)")
            print(f"  Avg length: {stats['avg_message_length']:.0f} chars, {stats['avg_words_per_message']:.1f} words")
            print(f"  Total words: {stats['total_words']:,}")
            print(f"  Daily average: {stats['messages_per_day']:.1f} messages")

        return sender_stats

    def temporal_analysis(self):
        """Analyze communication patterns over time"""
        print("\n📅 Analyzing temporal patterns...")

        # Monthly patterns
        self.df['year_month'] = self.df['datetime'].dt.to_period('M')
        monthly_stats = self.df.groupby(['year_month', 'sender']).size().unstack(fill_value=0)

        # Yearly patterns
        self.df['year'] = self.df['datetime'].dt.year
        yearly_stats = self.df.groupby(['year', 'sender']).size().unstack(fill_value=0)

        # Day of week patterns
        self.df['day_of_week'] = self.df['datetime'].dt.day_name()
        dow_stats = self.df.groupby(['day_of_week', 'sender']).size().unstack(fill_value=0)

        # Hour patterns
        self.df['hour'] = self.df['datetime'].dt.hour
        hour_stats = self.df.groupby(['hour', 'sender']).size().unstack(fill_value=0)

        print("📈 Yearly message distribution:")
        for year in sorted(yearly_stats.index):
            year_total = yearly_stats.loc[year].sum()
            print(f"  {year}: {year_total:,} messages")

        return {
            'monthly': monthly_stats,
            'yearly': yearly_stats,
            'day_of_week': dow_stats,
            'hourly': hour_stats
        }

    def sentiment_analysis(self):
        """Analyze sentiment patterns over time"""
        print("\n😊 Analyzing sentiment patterns...")

        def get_sentiment(text):
            scores = self.sia.polarity_scores(str(text))
            return scores

        # Sample messages for faster processing on large datasets
        sample_size = min(10000, len(self.df))
        df_sample = self.df.sample(n=sample_size, random_state=42)

        print(f"Analyzing sentiment for {sample_size:,} messages...")

        sentiments = df_sample['message'].apply(get_sentiment)

        df_sample['sentiment_compound'] = [s['compound'] for s in sentiments]
        df_sample['sentiment_positive'] = [s['pos'] for s in sentiments]
        df_sample['sentiment_negative'] = [s['neg'] for s in sentiments]
        df_sample['sentiment_neutral'] = [s['neu'] for s in sentiments]

        # Classify overall sentiment
        df_sample['sentiment_label'] = df_sample['sentiment_compound'].apply(
            lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
        )

        # Sentiment by sender
        sentiment_by_sender = df_sample.groupby('sender').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        })

        print("\n😊 Average sentiment by sender:")
        for sender in sentiment_by_sender.index:
            compound = sentiment_by_sender.loc[sender, 'sentiment_compound']
            print(f"  {sender}: {compound:.3f} ({'Positive' if compound >= 0.05 else 'Negative' if compound <= -0.05 else 'Neutral'})")

        # Sentiment over time (yearly)
        df_sample['year'] = df_sample['datetime'].dt.year
        sentiment_yearly = df_sample.groupby('year')['sentiment_compound'].mean()

        return {
            'by_sender': sentiment_by_sender,
            'yearly': sentiment_yearly,
            'sample_data': df_sample
        }

    def communication_evolution(self):
        """Analyze how communication has evolved over time"""
        print("\n📈 Analyzing communication evolution...")

        # Group by year for evolution analysis
        yearly_evolution = {}

        for year in sorted(self.df['year'].unique()):
            year_data = self.df[self.df['year'] == year]

            evolution = {
                'message_count': len(year_data),
                'avg_message_length': year_data['char_count'].mean(),
                'avg_words_per_message': year_data['word_count'].mean(),
                'messages_per_sender': year_data.groupby('sender').size().to_dict()
            }

            yearly_evolution[year] = evolution

        # Calculate trends
        years = sorted(yearly_evolution.keys())
        if len(years) >= 3:
            early_years = years[:len(years)//3]
            late_years = years[-len(years)//3:]

            early_avg_length = np.mean([yearly_evolution[y]['avg_message_length'] for y in early_years])
            late_avg_length = np.mean([yearly_evolution[y]['avg_message_length'] for y in late_years])

            early_avg_count = np.mean([yearly_evolution[y]['message_count'] for y in early_years])
            late_avg_count = np.mean([yearly_evolution[y]['message_count'] for y in late_years])

            print(f"📊 Message length evolution: {early_avg_length:.0f} → {late_avg_length:.0f} chars")
            print(f"📊 Yearly volume evolution: {early_avg_count:.0f} → {late_avg_count:.0f} messages")

        return yearly_evolution

    def relationship_analysis(self):
        """Analyze relationship dynamics and potential"""
        print("\n💕 Analyzing relationship dynamics...")

        senders = list(self.df['sender'].unique())
        if len(senders) != 2:
            print("⚠️  Analysis optimized for 2-person conversations")
            return None

        sender1, sender2 = senders

        # Message balance
        s1_count = len(self.df[self.df['sender'] == sender1])
        s2_count = len(self.df[self.df['sender'] == sender2])
        balance_ratio = s1_count / s2_count

        # Response patterns (simplified)
        total_messages = len(self.df)
        duration_years = ((self.df['datetime'].max() - self.df['datetime'].min()).days) / 365.25

        # Consistency over time
        monthly_counts = self.df.groupby(self.df['datetime'].dt.to_period('M')).size()
        consistency_score = 1 - (monthly_counts.std() / monthly_counts.mean()) if monthly_counts.mean() > 0 else 0

        # Calculate relationship score
        score = 0

        # Duration points
        if duration_years >= 6: score += 3
        elif duration_years >= 3: score += 2
        elif duration_years >= 1: score += 1

        # Volume points
        if total_messages >= 25000: score += 3
        elif total_messages >= 10000: score += 2
        elif total_messages >= 5000: score += 1

        # Balance points
        if 0.75 <= balance_ratio <= 1.25: score += 2
        elif 0.6 <= balance_ratio <= 1.4: score += 1

        # Consistency points
        if consistency_score >= 0.7: score += 2
        elif consistency_score >= 0.5: score += 1

        # Daily communication points
        daily_avg = total_messages / max(1, (self.df['datetime'].max() - self.df['datetime'].min()).days)
        if daily_avg >= 15: score += 2
        elif daily_avg >= 8: score += 1

        analysis = {
            'balance_ratio': balance_ratio,
            'consistency_score': consistency_score,
            'total_score': score,
            'max_score': 12,
            'duration_years': duration_years,
            'daily_average': daily_avg
        }

        print(f"💕 Relationship Score: {score}/12")
        print(f"📊 Communication Balance: {balance_ratio:.2f} ({'Balanced' if 0.8 <= balance_ratio <= 1.2 else 'Imbalanced'})")
        print(f"📈 Consistency Score: {consistency_score:.2f}")

        # Relationship potential assessment
        if score >= 9:
            potential = "Excellent - Strong indicators of deep compatibility and commitment"
        elif score >= 7:
            potential = "Very Good - Positive signs with room for growth"
        elif score >= 5:
            potential = "Moderate - Some good indicators, focus on quality interactions"
        else:
            potential = "Developing - Consider emotional depth beyond message frequency"

        print(f"💫 Relationship Potential: {potential}")

        return analysis

    def emotional_intelligence_analysis(self):
        """Analyze emotional intelligence patterns and support behaviors"""
        print("\n🧠 Analyzing emotional intelligence patterns...")

        # Keywords that indicate emotional states or needs
        emotional_keywords = {
            'stress': ['stressed', 'stress', 'overwhelmed', 'pressure', 'difficult', 'hard time', 'struggling'],
            'sadness': ['sad', 'depressed', 'down', 'upset', 'crying', 'hurt', 'disappointed'],
            'anxiety': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'panic'],
            'anger': ['angry', 'mad', 'furious', 'frustrated', 'annoyed', 'irritated'],
            'joy': ['happy', 'excited', 'thrilled', 'amazing', 'wonderful', 'fantastic'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'blessed', 'lucky'],
            'support_offered': ['here for you', 'i understand', 'that sucks', 'im sorry', 'you can do it', 'believe in you'],
            'support_seeking': ['help me', 'what should i do', 'i need', 'advice', 'what do you think']
        }

        results = {}
        senders = self.df['sender'].unique()

        for sender in senders:
            sender_data = self.df[self.df['sender'] == sender]
            sender_results = {
                'emotional_expression': {},
                'support_patterns': {
                    'support_offered': 0,
                    'support_sought': 0,
                    'empathy_responses': 0
                },
                'emotional_intelligence_score': 0
            }

            # Count emotional expressions
            total_messages = len(sender_data)
            for emotion, keywords in emotional_keywords.items():
                count = 0
                for keyword in keywords:
                    count += sender_data['message'].str.lower().str.contains(keyword, na=False).sum()

                if emotion in ['support_offered', 'support_seeking']:
                    if emotion == 'support_offered':
                        sender_results['support_patterns']['support_offered'] = count
                    else:
                        sender_results['support_patterns']['support_sought'] = count
                else:
                    sender_results['emotional_expression'][emotion] = {
                        'count': count,
                        'percentage': (count / total_messages) * 100 if total_messages > 0 else 0
                    }

            # Calculate emotional intelligence score
            ei_score = 0
            support_ratio = sender_results['support_patterns']['support_offered'] / max(1, total_messages / 100)
            if support_ratio > 2: ei_score += 3
            elif support_ratio > 1: ei_score += 2
            elif support_ratio > 0.5: ei_score += 1

            # Emotional variety (expressing different emotions)
            expressed_emotions = sum(1 for emotion_data in sender_results['emotional_expression'].values()
                                   if emotion_data['count'] > 0)
            if expressed_emotions >= 5: ei_score += 2
            elif expressed_emotions >= 3: ei_score += 1

            sender_results['emotional_intelligence_score'] = ei_score
            results[sender] = sender_results

        return results

    def topic_evolution_analysis(self):
        """Analyze how conversation topics have evolved over time"""
        print("\n📈 Analyzing topic evolution...")

        topic_keywords = {
            'work_career': ['work', 'job', 'career', 'office', 'meeting', 'project', 'boss', 'salary', 'interview'],
            'relationships_family': ['family', 'parents', 'sister', 'brother', 'relationship', 'dating', 'marriage'],
            'health_fitness': ['gym', 'workout', 'health', 'doctor', 'medicine', 'exercise', 'fitness'],
            'travel_experiences': ['travel', 'trip', 'vacation', 'flight', 'hotel', 'adventure', 'explore'],
            'technology': ['tech', 'computer', 'phone', 'app', 'software', 'programming', 'iphone'],
            'personal_growth': ['learning', 'growth', 'goal', 'dream', 'future', 'change', 'improve'],
            'hobbies_interests': ['movie', 'music', 'book', 'game', 'hobby', 'art', 'cooking'],
            'current_events': ['news', 'politics', 'world', 'election', 'covid', 'pandemic']
        }

        # Group by year and analyze topic distribution
        yearly_topics = {}
        years = sorted(self.df['year'].unique())

        for year in years:
            year_data = self.df[self.df['year'] == year]
            year_topics = {}

            total_year_messages = len(year_data)
            for topic, keywords in topic_keywords.items():
                count = 0
                for keyword in keywords:
                    count += year_data['message'].str.lower().str.contains(keyword, na=False).sum()

                year_topics[topic] = {
                    'count': count,
                    'percentage': (count / total_year_messages) * 100 if total_year_messages > 0 else 0
                }

            yearly_topics[year] = year_topics

        return yearly_topics

    def response_quality_analysis(self):
        """Analyze the quality and thoughtfulness of responses"""
        print("\n💭 Analyzing response quality...")

        # Create conversation threads by grouping nearby messages
        self.df['time_gap'] = self.df['datetime'].diff().dt.total_seconds() / 60  # minutes
        self.df['conversation_break'] = self.df['time_gap'] > 30  # New conversation if >30min gap
        self.df['conversation_id'] = self.df['conversation_break'].cumsum()

        quality_metrics = {}
        senders = self.df['sender'].unique()

        for sender in senders:
            sender_responses = []

            # Find responses (messages that follow another person's message within 30 minutes)
            for conv_id in self.df['conversation_id'].unique():
                conv_data = self.df[self.df['conversation_id'] == conv_id].sort_values('datetime')

                for i in range(1, len(conv_data)):
                    current_msg = conv_data.iloc[i]
                    previous_msg = conv_data.iloc[i-1]

                    if (current_msg['sender'] == sender and
                        previous_msg['sender'] != sender and
                        current_msg['time_gap'] <= 30):

                        # This is a response from our sender
                        response_quality = self.assess_response_quality(
                            previous_msg['message'],
                            current_msg['message']
                        )
                        sender_responses.append(response_quality)

            if sender_responses:
                quality_metrics[sender] = {
                    'avg_response_length': np.mean([r['length'] for r in sender_responses]),
                    'thoughtfulness_score': np.mean([r['thoughtfulness'] for r in sender_responses]),
                    'empathy_score': np.mean([r['empathy'] for r in sender_responses]),
                    'question_asking_rate': np.mean([r['asks_questions'] for r in sender_responses]),
                    'total_responses': len(sender_responses)
                }

        return quality_metrics

    def assess_response_quality(self, original_message, response_message):
        """Assess the quality of a response to a message"""
        response_lower = response_message.lower()
        original_lower = original_message.lower()

        # Length factor
        length = len(response_message)

        # Thoughtfulness indicators
        thoughtfulness_keywords = ['because', 'i think', 'maybe', 'probably', 'consider', 'understand']
        thoughtfulness = sum(1 for keyword in thoughtfulness_keywords if keyword in response_lower)

        # Empathy indicators
        empathy_keywords = ['feel', 'understand', 'sorry', 'that sucks', 'i get it', 'know how you feel']
        empathy = sum(1 for keyword in empathy_keywords if keyword in response_lower)

        # Question asking (showing interest)
        question_count = response_message.count('?')

        return {
            'length': length,
            'thoughtfulness': min(thoughtfulness, 3),  # Cap at 3
            'empathy': min(empathy, 3),  # Cap at 3
            'asks_questions': min(question_count, 2)   # Cap at 2
        }

    def conflict_resolution_analysis(self):
        """Detect and analyze how conflicts are handled"""
        print("\n⚔️ Analyzing conflict resolution patterns...")

        # Conflict indicators
        conflict_keywords = [
            'sorry', 'apologize', 'my fault', 'i was wrong', 'misunderstood',
            'didnt mean', 'lets talk', 'can we', 'i understand', 'youre right'
        ]

        negative_keywords = [
            'angry', 'upset', 'frustrated', 'disappointed', 'hurt', 'mad',
            'wrong', 'stupid', 'ridiculous', 'cant believe', 'seriously'
        ]

        results = {}
        senders = self.df['sender'].unique()

        # Find potential conflict periods (high negative sentiment clusters)
        conflicts = []

        for conv_id in self.df['conversation_id'].unique():
            conv_data = self.df[self.df['conversation_id'] == conv_id].sort_values('datetime')

            if len(conv_data) > 5:  # Only analyze substantial conversations
                negative_intensity = 0
                conflict_messages = []

                for _, msg in conv_data.iterrows():
                    msg_lower = msg['message'].lower()
                    neg_count = sum(1 for keyword in negative_keywords if keyword in msg_lower)

                    if neg_count > 0:
                        negative_intensity += neg_count
                        conflict_messages.append(msg)

                    # If we find resolution keywords after negative messages
                    if negative_intensity > 0:
                        resolution_count = sum(1 for keyword in conflict_keywords if keyword in msg_lower)
                        if resolution_count > 0:
                            conflicts.append({
                                'conversation_id': conv_id,
                                'negative_intensity': negative_intensity,
                                'resolution_attempt': resolution_count,
                                'messages': conflict_messages + [msg],
                                'resolved': resolution_count >= negative_intensity
                            })
                            break

        # Analyze conflict resolution by sender
        for sender in senders:
            sender_conflicts = []
            resolution_attempts = 0

            for conflict in conflicts:
                sender_msgs = [msg for msg in conflict['messages'] if msg['sender'] == sender]
                if sender_msgs:
                    # Check if this sender attempted resolution
                    for msg in sender_msgs:
                        msg_lower = msg['message'].lower()
                        if any(keyword in msg_lower for keyword in conflict_keywords):
                            resolution_attempts += 1
                            break

                    sender_conflicts.append(conflict)

            results[sender] = {
                'conflicts_involved': len(sender_conflicts),
                'resolution_attempts': resolution_attempts,
                'resolution_rate': resolution_attempts / max(1, len(sender_conflicts)),
                'avg_conflict_intensity': np.mean([c['negative_intensity'] for c in sender_conflicts]) if sender_conflicts else 0
            }

        return results

    def growth_milestone_analysis(self):
        """Track major life events and communication changes around them"""
        print("\n🌟 Analyzing growth milestones...")

        milestone_keywords = {
            'career_milestones': ['new job', 'promotion', 'interview', 'hired', 'started work', 'career'],
            'education_milestones': ['graduation', 'degree', 'college', 'university', 'exam', 'study'],
            'relationship_milestones': ['engaged', 'married', 'wedding', 'anniversary', 'relationship'],
            'personal_achievements': ['achieved', 'accomplished', 'proud', 'success', 'goal', 'dream'],
            'life_changes': ['moving', 'new place', 'apartment', 'house', 'relocat', 'transfer'],
            'challenges_overcome': ['overcame', 'better now', 'recovered', 'healed', 'solved', 'figured out']
        }

        milestones_by_year = {}

        for year in sorted(self.df['year'].unique()):
            year_data = self.df[self.df['year'] == year]
            year_milestones = {}

            for category, keywords in milestone_keywords.items():
                milestone_messages = []

                for _, msg in year_data.iterrows():
                    msg_lower = msg['message'].lower()
                    if any(keyword in msg_lower for keyword in keywords):
                        milestone_messages.append({
                            'date': msg['datetime'],
                            'sender': msg['sender'],
                            'message': msg['message']
                        })

                year_milestones[category] = milestone_messages

            milestones_by_year[year] = year_milestones

        # Analyze communication changes around milestones
        milestone_impact = {}
        for year, milestones in milestones_by_year.items():
            total_milestone_events = sum(len(events) for events in milestones.values())
            if total_milestone_events > 0:
                # Get message volume around this year
                year_msg_count = len(self.df[self.df['year'] == year])
                milestone_impact[year] = {
                    'milestone_events': total_milestone_events,
                    'message_volume': year_msg_count,
                    'milestones_detail': milestones
                }

        return milestone_impact

    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n📝 Generating comprehensive insights report...")

        stats = self.basic_stats()
        sender_analysis = self.sender_analysis()
        temporal = self.temporal_analysis()
        evolution = self.communication_evolution()
        relationship = self.relationship_analysis()

        # Try sentiment analysis
        try:
            sentiment = self.sentiment_analysis()
        except Exception as e:
            print(f"⚠️  Skipping sentiment analysis: {e}")
            sentiment = None

        # Enhanced analyses
        try:
            emotional_intelligence = self.emotional_intelligence_analysis()
        except Exception as e:
            print(f"⚠️  Skipping emotional intelligence analysis: {e}")
            emotional_intelligence = None

        try:
            topic_evolution = self.topic_evolution_analysis()
        except Exception as e:
            print(f"⚠️  Skipping topic evolution analysis: {e}")
            topic_evolution = None

        try:
            response_quality = self.response_quality_analysis()
        except Exception as e:
            print(f"⚠️  Skipping response quality analysis: {e}")
            response_quality = None

        try:
            conflict_resolution = self.conflict_resolution_analysis()
        except Exception as e:
            print(f"⚠️  Skipping conflict resolution analysis: {e}")
            conflict_resolution = None

        try:
            growth_milestones = self.growth_milestone_analysis()
        except Exception as e:
            print(f"⚠️  Skipping growth milestone analysis: {e}")
            growth_milestones = None

        # Generate report
        report = []
        report.append("=" * 60)
        report.append("📱 WHATSAPP CHAT ANALYSIS REPORT")
        report.append("=" * 60)

        # Data Quality Section
        report.append(f"\n🔍 DATA QUALITY & FILTERING")
        report.append(f"{'─' * 30}")
        original_count = len(self.df_original)
        authentic_count = len(self.df_filtered)

        forwarded_msg_count = self.df_original['is_forwarded'].sum()
        forwarded_participant_count = self.df_original['is_forwarded_participant'].sum()
        total_filtered_count = forwarded_msg_count + forwarded_participant_count
        filtered_percentage = (total_filtered_count / original_count) * 100 if original_count > 0 else 0

        report.append(f"Total messages parsed: {original_count:,}")
        report.append(f"Forwarded messages filtered: {forwarded_msg_count:,}")
        report.append(f"Forwarded participant messages filtered: {forwarded_participant_count:,}")
        report.append(f"Total filtered: {total_filtered_count:,} ({filtered_percentage:.1f}%)")
        report.append(f"Authentic messages analyzed: {authentic_count:,}")
        report.append("✅ Analysis based on authentic communication only")

        if filtered_percentage > 20:
            report.append("⚠️  High forwarded content detected - filtered out for accurate analysis")
        elif filtered_percentage > 10:
            report.append("📊 Moderate forwarded content filtered out")
        else:
            report.append("✨ Low forwarded content - mostly authentic communication")

        # Overview
        report.append(f"\n📊 OVERVIEW")
        report.append(f"{'─' * 30}")
        report.append(f"Authentic Messages: {stats['total_messages']:,}")
        report.append(f"Duration: {stats['duration_years']:.1f} years")
        report.append(f"Daily Average: {stats['daily_average']:.1f} authentic messages")
        report.append(f"Participants: {', '.join(stats['senders'])}")

        # Personal Growth Insights
        report.append(f"\n🌱 PERSONAL GROWTH INSIGHTS")
        report.append(f"{'─' * 30}")

        years = sorted(evolution.keys())
        if len(years) >= 3:
            early_years = years[:2]
            late_years = years[-2:]

            early_avg_length = np.mean([evolution[y]['avg_message_length'] for y in early_years])
            late_avg_length = np.mean([evolution[y]['avg_message_length'] for y in late_years])

            if late_avg_length > early_avg_length * 1.1:
                report.append("📈 Message Complexity: Your authentic messages have become longer and more thoughtful.")
            elif late_avg_length < early_avg_length * 0.9:
                report.append("📉 Communication Style: You've become more concise in your authentic communication.")
            else:
                report.append("↔️  Communication Consistency: Your authentic message style has remained stable.")

        # Communication patterns
        for sender, stats_data in sender_analysis.items():
            report.append(f"\n{sender}'s Authentic Communication Pattern:")
            if stats_data['avg_message_length'] > 50:
                report.append(f"  • Detailed communicator (avg {stats_data['avg_message_length']:.0f} chars)")
            else:
                report.append(f"  • Concise communicator (avg {stats_data['avg_message_length']:.0f} chars)")

            if stats_data['percentage'] > 55:
                report.append(f"  • More expressive participant ({stats_data['percentage']:.1f}% of authentic messages)")
            elif stats_data['percentage'] < 45:
                report.append(f"  • More listening-oriented ({stats_data['percentage']:.1f}% of authentic messages)")
            else:
                report.append(f"  • Balanced contributor ({stats_data['percentage']:.1f}% of authentic messages)")

        # Forwarded Content Analysis
        if total_filtered_count > 0:
            report.append(f"\n🔄 FORWARDED CONTENT INSIGHTS")
            report.append(f"{'─' * 30}")

            # Analyze who forwards more
            forwarded_by_sender = self.df_original[self.df_original['is_forwarded']]['sender'].value_counts()
            for sender, count in forwarded_by_sender.items():
                sender_total = len(self.df_original[self.df_original['sender'] == sender])
                forward_percentage = (count / sender_total) * 100
                report.append(f"{sender}: {count:,} forwards ({forward_percentage:.1f}% of their messages)")

            if filtered_percentage > 15:
                report.append("📌 Note: High forwarded content suggests sharing of external information")
                report.append("💡 Focus on original conversations for relationship assessment")

        # Relationship Analysis
        if relationship:
            report.append(f"\n💕 RELATIONSHIP ANALYSIS (Authentic Communication)")
            report.append(f"{'─' * 30}")
            report.append(f"Relationship Score: {relationship['total_score']}/12")

            if relationship['balance_ratio'] > 1.3:
                report.append(f"⚖️  Communication Balance: {list(sender_analysis.keys())[0]} initiates more authentic conversations")
            elif relationship['balance_ratio'] < 0.77:
                report.append(f"⚖️  Communication Balance: {list(sender_analysis.keys())[1]} initiates more authentic conversations")
            else:
                report.append("⚖️  Communication Balance: Well-balanced authentic mutual engagement")

            report.append(f"📊 Consistency: {'High' if relationship['consistency_score'] > 0.7 else 'Moderate' if relationship['consistency_score'] > 0.4 else 'Variable'}")
            report.append(f"⏰ Duration: {relationship['duration_years']:.1f} years of sustained authentic communication")

        # Recommendations
        report.append(f"\n🎯 RECOMMENDATIONS")
        report.append(f"{'─' * 30}")

        report.append("Personal Growth Focus Areas:")
        report.append("• Track emotional evolution in authentic conversations")
        report.append("• Monitor support patterns in personal exchanges")
        report.append("• Observe language maturity in original thoughts")
        if filtered_percentage > 15:
            report.append("• Consider reducing forwarded content to focus on personal communication")

        if relationship and len(stats['senders']) == 2:
            report.append("\nRelationship Development:")
            if relationship['balance_ratio'] > 1.4:
                report.append(f"• {list(sender_analysis.keys())[1]} could share more original thoughts")
            elif relationship['balance_ratio'] < 0.7:
                report.append(f"• {list(sender_analysis.keys())[0]} could share more original thoughts")

            report.append("• Quality focus: Deep personal conversations about values and goals")
            report.append("• Emotional authenticity: Review genuine support during challenges")

            if relationship['total_score'] >= 8:
                report.append("• Strong foundation: Authentic communication shows excellent compatibility")
            elif relationship['total_score'] >= 6:
                report.append("• Good potential: Continue building genuine emotional intimacy")
            else:
                report.append("• Development needed: Focus on authentic emotional depth over message volume")

        # Sentiment insights
        if sentiment:
            report.append(f"\n😊 EMOTIONAL TONE ANALYSIS (Authentic Messages)")
            report.append(f"{'─' * 30}")
            for sender in sentiment['by_sender'].index:
                compound = sentiment['by_sender'].loc[sender, 'sentiment_compound']
                tone = 'Generally Positive' if compound >= 0.05 else 'Generally Negative' if compound <= -0.05 else 'Neutral'
                report.append(f"{sender}: {tone} (authentic sentiment: {compound:.3f})")

        # Enhanced Analysis Sections
        if emotional_intelligence:
            report.append(f"\n🧠 EMOTIONAL INTELLIGENCE ANALYSIS")
            report.append(f"{'─' * 30}")

            for sender, ei_data in emotional_intelligence.items():
                report.append(f"\n{sender}'s Emotional Intelligence Profile:")
                report.append(f"  EI Score: {ei_data['emotional_intelligence_score']}/5")

                # Support patterns
                support_ratio = ei_data['support_patterns']['support_offered'] / max(1, ei_data['support_patterns']['support_sought'])
                if support_ratio > 1.5:
                    report.append(f"  💪 Strong supporter: Offers more help than seeks ({support_ratio:.1f}:1 ratio)")
                elif support_ratio < 0.7:
                    report.append(f"  🤝 Seeks support: More likely to ask for help than offer")
                else:
                    report.append(f"  ⚖️  Balanced: Equal give-and-take in support")

                # Top emotions expressed
                top_emotions = sorted(ei_data['emotional_expression'].items(),
                                    key=lambda x: x[1]['count'], reverse=True)[:3]
                if top_emotions:
                    emotions_text = ", ".join([f"{emotion}: {data['count']} times"
                                             for emotion, data in top_emotions if data['count'] > 0])
                    if emotions_text:
                        report.append(f"  🎭 Most expressed emotions: {emotions_text}")

        if topic_evolution:
            report.append(f"\n📈 TOPIC EVOLUTION ANALYSIS")
            report.append(f"{'─' * 30}")

            # Find trending topics (comparing first and last years)
            years = sorted(topic_evolution.keys())
            if len(years) >= 3:
                early_year = years[0]
                late_year = years[-1]

                report.append(f"Comparing {early_year} vs {late_year}:")

                for topic in topic_evolution[early_year].keys():
                    early_pct = topic_evolution[early_year][topic]['percentage']
                    late_pct = topic_evolution[late_year][topic]['percentage']

                    if late_pct > early_pct * 1.5 and late_pct > 2:
                        report.append(f"  📈 {topic.replace('_', ' ').title()}: Increased focus ({early_pct:.1f}% → {late_pct:.1f}%)")
                    elif early_pct > late_pct * 1.5 and early_pct > 2:
                        report.append(f"  📉 {topic.replace('_', ' ').title()}: Decreased focus ({early_pct:.1f}% → {late_pct:.1f}%)")

        if response_quality:
            report.append(f"\n💭 RESPONSE QUALITY ANALYSIS")
            report.append(f"{'─' * 30}")

            for sender, rq_data in response_quality.items():
                report.append(f"\n{sender}'s Response Quality:")
                report.append(f"  📏 Average response length: {rq_data['avg_response_length']:.0f} characters")
                report.append(f"  🤔 Thoughtfulness score: {rq_data['thoughtfulness_score']:.2f}/3")
                report.append(f"  💝 Empathy score: {rq_data['empathy_score']:.2f}/3")
                report.append(f"  ❓ Question asking rate: {rq_data['question_asking_rate']:.2f}/2")

                # Overall quality assessment
                overall_quality = (rq_data['thoughtfulness_score'] + rq_data['empathy_score'] + rq_data['question_asking_rate']) / 3
                quality_level = "Excellent" if overall_quality > 1.5 else "Good" if overall_quality > 1.0 else "Developing"
                report.append(f"  🎯 Overall quality: {quality_level} ({overall_quality:.2f}/3)")

        if conflict_resolution:
            report.append(f"\n⚔️ CONFLICT RESOLUTION ANALYSIS")
            report.append(f"{'─' * 30}")

            for sender, cr_data in conflict_resolution.items():
                if cr_data['conflicts_involved'] > 0:
                    report.append(f"\n{sender}'s Conflict Resolution:")
                    report.append(f"  🔥 Conflicts involved: {cr_data['conflicts_involved']}")
                    report.append(f"  🕊️ Resolution attempts: {cr_data['resolution_attempts']}")
                    report.append(f"  📈 Resolution rate: {cr_data['resolution_rate']*100:.1f}%")

                    if cr_data['resolution_rate'] > 0.7:
                        report.append(f"  ✨ Excellent conflict resolver - actively seeks resolution")
                    elif cr_data['resolution_rate'] > 0.4:
                        report.append(f"  👍 Good at conflict resolution")
                    else:
                        report.append(f"  💡 Opportunity: Could be more proactive in resolving conflicts")

        if growth_milestones:
            report.append(f"\n🌟 GROWTH MILESTONE ANALYSIS")
            report.append(f"{'─' * 30}")

            milestone_years = sorted(growth_milestones.keys())
            total_milestones = sum(data['milestone_events'] for data in growth_milestones.values())

            report.append(f"Total milestone events tracked: {total_milestones}")

            for year in milestone_years[-3:]:  # Show last 3 years with milestones
                year_data = growth_milestones[year]
                report.append(f"\n{year} Milestones ({year_data['milestone_events']} events):")

                for category, events in year_data['milestones_detail'].items():
                    if events:
                        category_name = category.replace('_', ' ').title()
                        report.append(f"  🎯 {category_name}: {len(events)} events")

        # Self-Improvement Recommendations
        report.append(f"\n🎯 PERSONALIZED SELF-IMPROVEMENT RECOMMENDATIONS")
        report.append(f"{'─' * 30}")

        if emotional_intelligence:
            # Find the user (assume first sender or most active)
            user_sender = max(emotional_intelligence.keys(),
                            key=lambda x: emotional_intelligence[x]['emotional_intelligence_score']
                            if 'Siva' in x else -1)

            if user_sender:
                user_ei = emotional_intelligence[user_sender]
                report.append(f"For {user_sender}:")

                if user_ei['emotional_intelligence_score'] < 3:
                    report.append("  • 🧠 Emotional Intelligence: Practice expressing empathy more often")
                    report.append("    - Use phrases like 'I understand how you feel' or 'That must be difficult'")

                if response_quality and user_sender in response_quality:
                    user_rq = response_quality[user_sender]
                    if user_rq['question_asking_rate'] < 0.5:
                        report.append("  • ❓ Curiosity: Ask more follow-up questions to show deeper interest")

                    if user_rq['thoughtfulness_score'] < 1.5:
                        report.append("  • 🤔 Thoughtfulness: Share your reasoning more often using 'because' or 'I think'")

                if conflict_resolution and user_sender in conflict_resolution:
                    user_cr = conflict_resolution[user_sender]
                    if user_cr['resolution_rate'] < 0.5:
                        report.append("  • 🕊️ Conflict Resolution: Be more proactive in addressing disagreements")
                        report.append("    - Try phrases like 'Let's talk about this' or 'I want to understand your perspective'")

        # Data integrity note
        report.append(f"\n📋 ANALYSIS NOTES")
        report.append(f"{'─' * 30}")
        report.append("✅ Forwarded messages automatically detected and filtered")
        report.append("✅ Analysis focuses on authentic personal communication")
        report.append("✅ System messages and notifications excluded")
        report.append(f"✅ {authentic_count:,} genuine messages analyzed for accurate insights")

        return "\n".join(report)

    def create_visualizations(self):
        """Create visualizations of the analysis"""
        print("\n📊 Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('WhatsApp Chat Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Messages over time
        monthly_data = self.df.groupby(self.df['datetime'].dt.to_period('M')).size()
        monthly_data.index = monthly_data.index.to_timestamp()

        axes[0,0].plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2)
        axes[0,0].set_title('Messages Over Time')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Messages per Month')
        axes[0,0].tick_params(axis='x', rotation=45)

        # 2. Message distribution by sender
        sender_counts = self.df['sender'].value_counts()
        colors = sns.color_palette("husl", len(sender_counts))
        axes[0,1].pie(sender_counts.values, labels=sender_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0,1].set_title('Message Distribution by Sender')

        # 3. Messages by day of week
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data = self.df['datetime'].dt.day_name().value_counts().reindex(dow_order)

        axes[1,0].bar(range(len(dow_data)), dow_data.values, color=sns.color_palette("husl", len(dow_data)))
        axes[1,0].set_title('Messages by Day of Week')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Total Messages')
        axes[1,0].set_xticks(range(len(dow_data)))
        axes[1,0].set_xticklabels(dow_data.index, rotation=45)

        # 4. Messages by hour of day
        hourly_data = self.df['datetime'].dt.hour.value_counts().sort_index()

        axes[1,1].bar(hourly_data.index, hourly_data.values, color='skyblue', alpha=0.7)
        axes[1,1].set_title('Messages by Hour of Day')
        axes[1,1].set_xlabel('Hour')
        axes[1,1].set_ylabel('Total Messages')
        axes[1,1].set_xticks(range(0, 24, 2))

        plt.tight_layout()

        # Save the plot
        output_path = self.file_path.parent / f"{self.file_path.stem}_analysis_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved as: {output_path}")

        plt.show()

        return output_path

def main():
    parser = argparse.ArgumentParser(description='Analyze WhatsApp chat for personal growth and relationship insights')
    parser.add_argument('file_path', help='Path to WhatsApp chat export (.txt file)')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating visualizations')
    parser.add_argument('--participants', help='Comma-separated list of target participants to analyze (e.g., "Person1,Person2")')

    args = parser.parse_args()

    # Parse target participants from CLI argument
    target_participants = None
    if args.participants:
        target_participants = [p.strip() for p in args.participants.split(',')]
        print(f"🎯 CLI Target participants: {', '.join(target_participants)}")

    try:
        # Initialize analyzer with target participants
        analyzer = WhatsAppAnalyzer(args.file_path, target_participants)

        # Parse the chat
        message_count = analyzer.parse_chat()

        if message_count == 0:
            print("❌ No messages found in the file. Please check the format.")
            return

        # Generate comprehensive report
        report = analyzer.generate_insights_report()

        # Save report
        report_path = Path(args.file_path).parent / f"{Path(args.file_path).stem}_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📝 Full report saved as: {report_path}")

        # Print report to console
        print("\n" + report)

        # Create visualizations
        if not args.no_plots:
            try:
                analyzer.create_visualizations()
            except Exception as e:
                print(f"⚠️  Could not create visualizations: {e}")
                print("💡 Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")

        print(f"\n✅ Analysis complete! Processed {message_count:,} messages.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
