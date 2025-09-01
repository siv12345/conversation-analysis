# Additional methods to enhance your WhatsApp analyzer

def vulnerability_analysis(self):
    """Analyze vulnerability and openness in conversations"""
    print("\n💖 Analyzing vulnerability and emotional openness...")
    
    vulnerability_keywords = {
        'personal_sharing': ['i feel', 'i struggle with', 'i worry about', 'i dream of', 'i hope', 'i fear'],
        'admitting_mistakes': ['i was wrong', 'my mistake', 'i messed up', 'i should have', 'i regret'],
        'seeking_advice': ['what do you think', 'help me decide', 'i dont know what to do', 'advice'],
        'sharing_insecurities': ['insecure', 'not good enough', 'worried about', 'scared that'],
        'expressing_needs': ['i need', 'i wish', 'could you', 'would you mind'],
        'deep_questions': ['do you ever', 'what if', 'how do you feel about', 'what matters to you']
    }
    
    results = {}
    senders = self.df['sender'].unique()
    
    for sender in senders:
        sender_data = self.df[self.df['sender'] == sender]
        total_messages = len(sender_data)
        
        vulnerability_scores = {}
        total_vulnerability = 0
        
        for category, keywords in vulnerability_keywords.items():
            count = 0
            for keyword in keywords:
                count += sender_data['message'].str.lower().str.contains(keyword, na=False).sum()
            
            percentage = (count / total_messages) * 100 if total_messages > 0 else 0
            vulnerability_scores[category] = {
                'count': count,
                'percentage': percentage
            }
            total_vulnerability += count
        
        # Calculate overall vulnerability score (0-10 scale)
        vulnerability_score = min(10, (total_vulnerability / max(1, total_messages / 10)))
        
        results[sender] = {
            'vulnerability_score': vulnerability_score,
            'categories': vulnerability_scores,
            'total_vulnerable_expressions': total_vulnerability
        }
    
    return results

def communication_maturity_analysis(self):
    """Analyze maturity in communication patterns"""
    print("\n🎯 Analyzing communication maturity...")
    
    maturity_indicators = {
        'active_listening': ['tell me more', 'i understand', 'that makes sense', 'i hear you'],
        'constructive_feedback': ['maybe consider', 'what about', 'another way to look at', 'perspective'],
        'emotional_regulation': ['lets talk later', 'i need time to think', 'cooling off', 'when im calmer'],
        'appreciation': ['thank you for', 'i appreciate', 'grateful for', 'means a lot'],
        'boundary_setting': ['i need space', 'not comfortable with', 'prefer not to', 'boundaries'],
        'future_planning': ['lets plan', 'looking forward to', 'goal is', 'working towards']
    }
    
    results = {}
    senders = self.df['sender'].unique()
    
    for sender in senders:
        sender_data = self.df[self.df['sender'] == sender]
        total_messages = len(sender_data)
        
        maturity_scores = {}
        total_maturity_indicators = 0
        
        for category, keywords in maturity_indicators.items():
            count = 0
            for keyword in keywords:
                count += sender_data['message'].str.lower().str.contains(keyword, na=False).sum()
            
            percentage = (count / total_messages) * 100 if total_messages > 0 else 0
            maturity_scores[category] = {
                'count': count,
                'percentage': percentage
            }
            total_maturity_indicators += count
        
        # Calculate maturity score (0-10 scale)
        maturity_score = min(10, (total_maturity_indicators / max(1, total_messages / 15)))
        
        results[sender] = {
            'maturity_score': maturity_score,
            'categories': maturity_scores,
            'total_maturity_indicators': total_maturity_indicators
        }
    
    return results

def relationship_depth_analysis(self):
    """Analyze depth of relationship through conversation topics"""
    print("\n🔍 Analyzing relationship depth...")
    
    depth_levels = {
        'surface_level': ['weather', 'food', 'tv show', 'movie', 'work', 'busy'],
        'personal_interests': ['hobby', 'passion', 'enjoy doing', 'love to', 'interest', 'favorite'],
        'values_beliefs': ['believe in', 'important to me', 'value', 'principle', 'meaning', 'purpose'],
        'future_dreams': ['dream', 'goal', 'future', 'hope to', 'someday', 'planning'],
        'fears_concerns': ['worried about', 'afraid of', 'scared', 'anxiety', 'concern', 'fear'],
        'relationship_meta': ['our relationship', 'us', 'we are', 'between us', 'how we', 'our future']
    }
    
    # Calculate conversation depth over time
    yearly_depth = {}
    
    for year in sorted(self.df['year'].unique()):
        year_data = self.df[self.df['year'] == year]
        total_year_messages = len(year_data)
        
        depth_scores = {}
        for level, keywords in depth_levels.items():
            count = 0
            for keyword in keywords:
                count += year_data['message'].str.lower().str.contains(keyword, na=False).sum()
            
            depth_scores[level] = (count / total_year_messages) * 100 if total_year_messages > 0 else 0
        
        # Calculate overall depth score for the year
        surface_weight = 0.5
        personal_weight = 1.0
        values_weight = 2.0
        future_weight = 1.5
        fears_weight = 2.5
        meta_weight = 3.0
        
        overall_depth = (
            depth_scores['surface_level'] * surface_weight +
            depth_scores['personal_interests'] * personal_weight +
            depth_scores['values_beliefs'] * values_weight +
            depth_scores['future_dreams'] * future_weight +
            depth_scores['fears_concerns'] * fears_weight +
            depth_scores['relationship_meta'] * meta_weight
        ) / 10  # Normalize
        
        yearly_depth[year] = {
            'depth_categories': depth_scores,
            'overall_depth_score': min(10, overall_depth)
        }
    
    return yearly_depth

def personal_growth_tracking(self):
    """Track specific personal growth indicators"""
    print("\n🌱 Tracking personal growth indicators...")
    
    growth_indicators = {
        'self_awareness': ['i realize', 'i learned', 'i notice', 'i understand now', 'makes me think'],
        'accountability': ['my fault', 'i caused', 'i take responsibility', 'i own that', 'my mistake'],
        'goal_setting': ['goal is', 'working on', 'trying to improve', 'want to be better', 'challenge myself'],
        'learning_mindset': ['learned from', 'lesson', 'experience taught me', 'now i know', 'mistake'],
        'empathy_growth': ['understand how you feel', 'see your point', 'perspective', 'in your shoes'],
        'emotional_intelligence': ['feeling', 'emotion', 'react', 'trigger', 'calm down', 'emotional']
    }
    
    # Track growth over time periods
    growth_timeline = {}
    years = sorted(self.df['year'].unique())
    
    for i, year in enumerate(years):
        year_data = self.df[self.df['year'] == year]
        total_messages = len(year_data)
        
        growth_scores = {}
        for indicator, keywords in growth_indicators.items():
            count = 0
            for keyword in keywords:
                count += year_data['message'].str.lower().str.contains(keyword, na=False).sum()
            
            growth_scores[indicator] = (count / total_messages) * 100 if total_messages > 0 else 0
        
        # Calculate growth trajectory (compare to previous years if available)
        if i > 0:
            prev_year = years[i-1]
            improvement_indicators = []
            
            for indicator in growth_indicators.keys():
                prev_score = growth_timeline.get(prev_year, {}).get('growth_categories', {}).get(indicator, 0)
                current_score = growth_scores[indicator]
                
                if current_score > prev_score * 1.2:  # 20% improvement
                    improvement_indicators.append(indicator)
        else:
            improvement_indicators = []
        
        growth_timeline[year] = {
            'growth_categories': growth_scores,
            'improvement_areas': improvement_indicators,
            'total_growth_score': sum(growth_scores.values()) / len(growth_scores)
        }
    
    return growth_timeline

def relationship_compatibility_deep_dive(self):
    """Enhanced relationship compatibility analysis"""
    print("\n💕 Deep relationship compatibility analysis...")
    
    if len(self.df['sender'].unique()) != 2:
        return None
    
    sender1, sender2 = self.df['sender'].unique()
    
    compatibility_factors = {
        'communication_sync': self._analyze_communication_sync(sender1, sender2),
        'emotional_support': self._analyze_mutual_support(sender1, sender2),
        'shared_interests': self._analyze_shared_topics(sender1, sender2),
        'conflict_style': self._analyze_conflict_compatibility(sender1, sender2),
        'future_alignment': self._analyze_future_discussions(sender1, sender2),
        'vulnerability_match': self._analyze_vulnerability_balance(sender1, sender2)
    }
    
    # Calculate overall compatibility score
    total_score = sum(factor['score'] for factor in compatibility_factors.values())
    max_possible = len(compatibility_factors) * 10
    
    compatibility_percentage = (total_score / max_possible) * 100
    
    return {
        'compatibility_percentage': compatibility_percentage,
        'detailed_factors': compatibility_factors,
        'relationship_stage': self._determine_relationship_stage(compatibility_percentage),
        'growth_recommendations': self._generate_compatibility_recommendations(compatibility_factors)
    }

def _analyze_communication_sync(self, sender1, sender2):
    """Analyze how well communication styles sync"""
    s1_data = self.df[self.df['sender'] == sender1]
    s2_data = self.df[self.df['sender'] == sender2]
    
    # Response time analysis (simplified)
    s1_avg_length = s1_data['char_count'].mean()
    s2_avg_length = s2_data['char_count'].mean()
    
    # Style similarity score
    length_ratio = min(s1_avg_length, s2_avg_length) / max(s1_avg_length, s2_avg_length)
    
    # Activity pattern similarity
    s1_hours = s1_data['datetime'].dt.hour.value_counts(normalize=True)
    s2_hours = s2_data['datetime'].dt.hour.value_counts(normalize=True)
    
    # Calculate correlation of activity patterns
    common_hours = set(s1_hours.index) & set(s2_hours.index)
    if len(common_hours) > 12:  # At least half the day overlap
        sync_score = length_ratio * 5 + 5  # Scale to 0-10
    else:
        sync_score = length_ratio * 3 + 2  # Lower score for poor time sync
    
    return {
        'score': min(10, sync_score),
        'details': f"Communication style match: {length_ratio:.2f}, Active hour overlap: {len(common_hours)}/24"
    }

def _analyze_mutual_support(self, sender1, sender2):
    """Analyze mutual emotional support patterns"""
    support_keywords = ['sorry', 'understand', 'here for you', 'support', 'comfort', 'care about']
    
    s1_support = sum(self.df[self.df['sender'] == sender1]['message'].str.lower().str.contains(keyword, na=False).sum() 
                    for keyword in support_keywords)
    s2_support = sum(self.df[self.df['sender'] == sender2]['message'].str.lower().str.contains(keyword, na=False).sum() 
                    for keyword in support_keywords)
    
    total_s1_messages = len(self.df[self.df['sender'] == sender1])
    total_s2_messages = len(self.df[self.df['sender'] == sender2])
    
    s1_support_rate = s1_support / max(1, total_s1_messages / 100)
    s2_support_rate = s2_support / max(1, total_s2_messages / 100)
    
    # Balance and overall level
    balance_score = min(s1_support_rate, s2_support_rate) / max(s1_support_rate, s2_support_rate) if max(s1_support_rate, s2_support_rate) > 0 else 0
    overall_level = (s1_support_rate + s2_support_rate) / 2
    
    support_score = (balance_score * 5) + min(5, overall_level)
    
    return {
        'score': min(10, support_score),
        'details': f"Support balance: {balance_score:.2f}, Overall support level: {overall_level:.2f}"
    }

def generate_personal_improvement_plan(self, analyses):
    """Generate a personalized improvement plan based on all analyses"""
    print("\n🎯 Generating personal improvement plan...")
    
    improvement_plan = {
        'priority_areas': [],
        'specific_actions': [],
        'tracking_metrics': [],
        'timeline_goals': {}
    }
    
    # Analyze areas needing improvement
    if 'vulnerability' in analyses:
        user_vulnerability = analyses['vulnerability']['your_score']  # Assuming you identify the user
        if user_vulnerability < 5:
            improvement_plan['priority_areas'].append('Emotional Openness')
            improvement_plan['specific_actions'].append('Share one personal struggle or fear each week')
    
    if 'maturity' in analyses:
        user_maturity = analyses['maturity']['your_score']
        if user_maturity < 6:
            improvement_plan['priority_areas'].append('Communication Maturity')
            improvement_plan['specific_actions'].append('Practice active listening by asking follow-up questions')
    
    # Add tracking metrics
    improvement_plan['tracking_metrics'] = [
        'Weekly vulnerability expressions count',
        'Monthly conflict resolution success rate',
        'Emotional support offering frequency'
    ]
    
    # Set timeline goals
    improvement_plan['timeline_goals'] = {
        '1_month': 'Increase personal sharing by 20%',
        '3_months': 'Improve empathy response rate to 80%',
        '6_months': 'Achieve balanced communication ratio (45-55%)',
        '1_year': 'Reach mature communication score of 7+'
    }
    
    return improvement_plan