# WhatsApp Chat Analyzer

A comprehensive Python tool for analyzing personal growth and relationship dynamics from WhatsApp chat exports. This tool provides insights into communication patterns, sentiment analysis, vulnerability expressions, and relationship dynamics.

## Features

- **Message Parsing**: Automatically parses WhatsApp chat export files
- **Sentiment Analysis**: Analyzes emotional tone of conversations using NLTK's VADER sentiment analyzer
- **Vulnerability Analysis**: Identifies emotional openness and personal sharing patterns
- **Communication Maturity**: Evaluates communication patterns and maturity levels
- **Relationship Dynamics**: Analyzes interaction patterns between participants
- **Statistical Reports**: Generates comprehensive text reports and visual dashboards
- **Participant Filtering**: Focus analysis on specific chat participants
- **Forwarded Message Detection**: Automatically filters out forwarded messages for accurate analysis

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Navigate to the project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies
- pandas: Data manipulation and analysis
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- numpy: Numerical computing
- nltk: Natural language processing
- textstat: Text readability statistics
- wordcloud: Word cloud generation

## Usage

### Basic Usage
```bash
python whatsapp_analyzer.py <chat_export_file>
```

### With Specific Participants
```bash
python whatsapp_analyzer.py chat.txt --participants "Person1,Person2"
```

### Configuration
You can configure target participants directly in the code by editing the `TARGET_PARTICIPANTS` list in `whatsapp_analyzer.py`:

```python
TARGET_PARTICIPANTS = ["A", "B"]  # Replace with actual names
```

## Preparing Your WhatsApp Chat Export

1. Open WhatsApp on your phone
2. Go to the chat you want to analyze
3. Tap the three dots menu → More → Export chat
4. Choose "Without Media" (recommended for faster processing)
5. Save the exported `.txt` file
6. Transfer the file to your computer where the analyzer is installed

## Output Files

The analyzer generates several output files:

- **Text Report** (`*_analysis_report.txt`): Comprehensive statistical analysis
- **Dashboard Image** (`*_analysis_dashboard.png`): Visual charts and graphs
- **Console Output**: Real-time analysis progress and key insights

## Analysis Categories

### 1. Message Statistics
- Total messages per participant
- Average message length
- Most active time periods
- Response time patterns

### 2. Sentiment Analysis
- Positive, negative, and neutral sentiment scores
- Emotional trend analysis over time
- Sentiment distribution by participant

### 3. Vulnerability Analysis
Categories analyzed include:
- Personal sharing ("I feel", "I struggle with")
- Admitting mistakes ("I was wrong", "My mistake")
- Seeking advice ("What do you think", "Help me decide")
- Sharing insecurities ("Insecure", "Not good enough")
- Expressing needs ("I need", "I wish")
- Deep questions ("Do you ever", "What if")

### 4. Communication Maturity
- Use of mature language patterns
- Conflict resolution approaches
- Empathy expressions
- Constructive communication indicators

### 5. Relationship Dynamics
- Interaction balance
- Initiative patterns
- Support expressions
- Engagement levels

## Understanding the Results

### Vulnerability Score (0-10 scale)
- **0-3**: Low vulnerability/openness
- **4-6**: Moderate vulnerability/openness  
- **7-10**: High vulnerability/openness

### Sentiment Scores
- **Positive**: 0.05 to 1.0 (higher = more positive)
- **Neutral**: -0.05 to 0.05
- **Negative**: -1.0 to -0.05 (lower = more negative)

### Communication Maturity Indicators
- Active listening expressions
- Empathy statements
- Constructive feedback patterns
- Conflict resolution language

## Privacy and Data Security

- **Local Processing**: All analysis is performed locally on your machine
- **No Data Upload**: Your chat data never leaves your computer
- **Temporary Files**: Generated reports can be deleted after viewing
- **Anonymization**: Consider using initials or codes instead of real names

## Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment if using one

**"File not found"**
- Check the file path is correct
- Ensure the chat export file is in the same directory or provide full path

**"No messages parsed"**
- Verify the chat export format is supported
- Check that the file contains actual WhatsApp messages
- Ensure proper encoding (UTF-8)

**Empty or incomplete analysis**
- Check if participant names in TARGET_PARTICIPANTS match the actual names in the chat
- Verify the chat contains sufficient messages for meaningful analysis

### File Format Requirements
- WhatsApp chat export format (.txt)
- UTF-8 encoding
- Standard WhatsApp timestamp format
- Minimum ~50-100 messages for meaningful analysis

## Advanced Usage

### Customizing Analysis
- Modify keyword lists in the analyzer classes to match your language/culture
- Adjust sentiment analysis parameters
- Add custom analysis categories by extending the analyzer classes

### Batch Processing
For analyzing multiple chats, create a script that iterates through multiple export files:

```python
import os
from whatsapp_analyzer import WhatsAppAnalyzer

chat_files = ['chat1.txt', 'chat2.txt', 'chat3.txt']
for chat_file in chat_files:
    analyzer = WhatsAppAnalyzer(chat_file)
    # Run analysis...
```

## Ethical Considerations

- **Consent**: Only analyze chats where all participants have given consent
- **Privacy**: Be mindful of sensitive information in the reports
- **Purpose**: Use insights constructively for relationship improvement
- **Storage**: Securely handle and dispose of generated reports

## Contributing

To extend the analyzer:
1. Add new analysis methods to the `WhatsAppAnalyzer` class
2. Update the reporting functions to include new metrics
3. Test with various chat formats and languages
4. Document new features and parameters

## License

This project is for personal and educational use. Respect privacy and obtain consent before analyzing others' conversations.