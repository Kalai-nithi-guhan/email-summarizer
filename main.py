"""
Flask API Backend for Email/PDF Summarizer
Handles file uploads and returns summarized results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import json

# Import your existing summarizer
import re
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter
import PyPDF2

# Your existing EmailSummarizer class (paste the entire class here)
class EmailSummarizer:
    """Professional Email/PDF Summarizer"""
    
    def __init__(self):
        self.urgent_keywords = {
            'urgent': 25, 'immediately': 25, 'asap': 25, 'critical': 22,
            'important': 18, 'priority': 18, 'must': 15, 'essential': 15,
            'mandatory': 20, 'required': 15, 'crucial': 22, 'vital': 18,
            'emergency': 25, 'pressing': 20
        }
        
        self.task_keywords = {
            'complete': 12, 'submit': 15, 'send': 12, 'prepare': 12,
            'review': 10, 'update': 10, 'finish': 12, 'deliver': 15,
            'ensure': 10, 'action': 12, 'provide': 10, 'create': 10,
            'develop': 10, 'implement': 10, 'execute': 12
        }
        
        self.deadline_keywords = {
            'deadline': 18, 'due': 18, 'by': 15, 'before': 12,
            'until': 12, 'end of': 15, 'no later than': 18,
            'expires': 15, 'final': 15, 'last day': 18
        }
        
        self.meeting_keywords = {
            'meeting': 12, 'call': 10, 'conference': 12, 'discussion': 10,
            'scheduled': 12, 'appointment': 12, 'session': 10,
            'presentation': 12, 'demo': 10, 'workshop': 10
        }
    
    def read_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_dates(self, text: str) -> List[str]:
        patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}\b',
            r'\b(?:today|tomorrow|this\s+(?:week|month)|next\s+(?:week|month|Monday|Tuesday|Wednesday|Thursday|Friday))\b'
        ]
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        return list(set(dates))
    
    def clean_and_split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'^\s*(?:Subject|From|To|Date|Cc|Bcc):.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if 20 <= len(sent) <= 500 and any(c.isalpha() for c in sent):
                cleaned.append(sent)
        return cleaned
    
    def calculate_keyword_score(self, text_lower: str, keywords_dict: dict) -> Tuple[int, int]:
        score = 0
        count = 0
        for keyword, weight in keywords_dict.items():
            if keyword in text_lower:
                score += weight
                count += 1
        return score, count
    
    def determine_type_and_urgency(self, text_lower: str, has_date: bool) -> Tuple[str, str]:
        urgent_score, _ = self.calculate_keyword_score(text_lower, self.urgent_keywords)
        task_score, _ = self.calculate_keyword_score(text_lower, self.task_keywords)
        deadline_score, _ = self.calculate_keyword_score(text_lower, self.deadline_keywords)
        meeting_score, _ = self.calculate_keyword_score(text_lower, self.meeting_keywords)
        
        scores = {
            'deadline': deadline_score,
            'task': task_score,
            'meeting': meeting_score,
            'important_info': 0
        }
        
        sent_type = max(scores, key=scores.get)
        if scores[sent_type] == 0:
            sent_type = 'important_info'
        
        if urgent_score > 18 or deadline_score > 15:
            urgency = 'high'
        elif task_score > 12 or meeting_score > 10 or has_date:
            urgency = 'medium'
        else:
            urgency = 'low'
        
        return sent_type, urgency
    
    def score_sentences(self, sentences: List[str], dates: List[str]) -> List[Dict]:
        scored = []
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            score = 0
            found_date = None
            
            for date in dates:
                if date.lower() in sent_lower:
                    score += 20
                    found_date = date
                    break
            
            urgent_score, _ = self.calculate_keyword_score(sent_lower, self.urgent_keywords)
            task_score, _ = self.calculate_keyword_score(sent_lower, self.task_keywords)
            deadline_score, _ = self.calculate_keyword_score(sent_lower, self.deadline_keywords)
            meeting_score, _ = self.calculate_keyword_score(sent_lower, self.meeting_keywords)
            
            score += urgent_score + task_score + deadline_score + meeting_score
            position_bonus = max(0, 12 - idx)
            score += position_bonus
            
            word_count = len(sent.split())
            if 10 <= word_count <= 30:
                score += 6
            elif 30 < word_count <= 40:
                score += 3
            
            number_count = len(re.findall(r'\d+', sent))
            score += min(number_count * 3, 9)
            
            action_verbs = ['submit', 'complete', 'send', 'prepare', 'review', 'update', 'attend', 'confirm', 'deliver', 'finish']
            action_count = sum(1 for verb in action_verbs if verb in sent_lower)
            score += action_count * 4
            
            if '!' in sent:
                score += 6
            
            sent_type, urgency = self.determine_type_and_urgency(sent_lower, found_date is not None)
            
            scored.append({
                'text': sent,
                'score': score,
                'type': sent_type,
                'urgency': urgency,
                'date': found_date,
                'word_count': word_count
            })
        
        return scored
    
    def get_top_unique_sentences(self, scored_sentences: List[Dict], n: int = 5) -> List[Dict]:
        sorted_sentences = sorted(scored_sentences, key=lambda x: x['score'], reverse=True)
        selected = []
        seen_content = set()
        
        for sent in sorted_sentences:
            simplified = re.sub(r'[^\w\s]', '', sent['text'].lower())
            words = set(simplified.split())
            
            is_duplicate = False
            for seen_words in seen_content:
                if len(words & seen_words) / len(words | seen_words) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected.append(sent)
                seen_content.add(frozenset(words))
            
            if len(selected) >= n:
                break
        
        return selected
    
    def generate_summary(self, top_sentences: List[Dict]) -> str:
        type_counts = Counter(s['type'] for s in top_sentences)
        urgency_counts = Counter(s['urgency'] for s in top_sentences)
        
        parts = []
        if type_counts['deadline'] > 0:
            parts.append(f"{type_counts['deadline']} deadline(s)")
        if type_counts['task'] > 0:
            parts.append(f"{type_counts['task']} task(s)")
        if type_counts['meeting'] > 0:
            parts.append(f"{type_counts['meeting']} meeting(s)")
        
        urgency_note = f" ({urgency_counts['high']} urgent)" if urgency_counts['high'] > 0 else ""
        
        return "Contains: " + ", ".join(parts) + urgency_note if parts else "Important information"
    
    def summarize(self, text: str) -> Dict:
        dates = self.extract_dates(text)
        sentences = self.clean_and_split_sentences(text)
        
        if len(sentences) == 0:
            return {
                "top_5_points": [],
                "overall_summary": "No valid content found",
                "metadata": {"error": "No sentences extracted"}
            }
        
        scored_sentences = self.score_sentences(sentences, dates)
        top_5 = self.get_top_unique_sentences(scored_sentences, 5)
        summary = self.generate_summary(top_5)
        
        points = []
        for i, sent in enumerate(top_5, 1):
            points.append({
                "rank": i,
                "type": sent['type'],
                "point": sent['text'].strip(),
                "date": sent['date'],
                "urgency": sent['urgency']
            })
        
        return {
            "top_5_points": points,
            "overall_summary": summary,
            "metadata": {
                "total_sentences": len(sentences),
                "dates_found": len(dates),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    
    def process_pdf(self, pdf_path: str) -> Dict:
        text = self.read_pdf(pdf_path)
        return self.summarize(text)


# Flask App Configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize summarizer
summarizer = EmailSummarizer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Email PDF Summarizer',
        'version': '1.0.0'
    })

@app.route('/api/summarize', methods=['POST'])
def summarize_pdf():
    """Main endpoint for PDF summarization"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process PDF
        result = summarizer.process_pdf(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Flask Backend Server Starting...")
    print("=" * 60)
    print("📡 API Endpoints:")
    print("   GET  /api/health    - Health check")
    print("   POST /api/summarize - PDF summarization")
    print("=" * 60)
    print("🌐 Server running on: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)