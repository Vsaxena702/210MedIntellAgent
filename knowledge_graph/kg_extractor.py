"""
Knowledge Graph Extractor for Medical Records
Extracts Subject-Predicate-Object triples from patient medical data
"""

from openai import OpenAI
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MedicalKGExtractor:
    """Extract knowledge graph triples from medical records using LLM"""
    
    # Medical extraction prompt
    EXTRACTION_SYSTEM_PROMPT = """
You are a medical knowledge graph extraction specialist. 
Your task is to extract Subject-Predicate-Object (SPO) triples from medical records.

Focus on capturing:
- Patient-condition relationships
- Medication-condition relationships  
- Procedure-condition relationships
- Temporal relationships (when things occurred)
- Causal relationships (what treats/causes what)

Be precise and use medical terminology appropriately.
"""
    
    EXTRACTION_USER_PROMPT = """
Extract Subject-Predicate-Object triples from the medical data below.

**CRITICAL RULES:**
1. Output ONLY valid JSON with key "triples" containing an array
2. Each triple MUST have "subject", "predicate", "object" keys
3. Use lowercase for all values
4. Keep predicates concise (1-3 words: "has", "treats", "prescribed", "diagnosed_with")
5. Replace pronouns with "patient"
6. Be specific (e.g., "metformin 500mg" not just "metformin")

**Medical Data:**
{medical_context}

**Required JSON Format:**
{{
  "triples": [
    {{"subject": "patient", "predicate": "has_condition", "object": "hypertension"}},
    {{"subject": "hypertension", "predicate": "treated_with", "object": "lisinopril 10mg"}}
  ]
}}

**Your JSON Output:**
"""
    
    def __init__(self, openai_client: OpenAI = None, model: str = "gpt-4o-mini"):
        """
        Initialize the extractor
        
        Args:
            openai_client: OpenAI client instance
            model: LLM model to use
        """
        self.client = openai_client
        self.model = model
        
    def extract_from_records(self, records: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract SPO triples from retrieved medical records
        
        Args:
            records: Dictionary containing medical data (medications, conditions, etc.)
            
        Returns:
            List of triples in format [{"subject": ..., "predicate": ..., "object": ...}]
        """
        try:
            # Format the medical context
            context = self._format_medical_context(records)
            
            if not context.strip():
                logger.warning("No medical context to extract from")
                return []
            
            # Call LLM to extract triples
            triples = self._call_llm_extraction(context)
            
            # Normalize and validate
            normalized = self._normalize_triples(triples)
            
            logger.info(f"Extracted {len(normalized)} valid triples")
            return normalized
            
        except Exception as e:
            logger.error(f"Error extracting triples: {e}")
            return []
    
    def _format_medical_context(self, records: Dict[str, Any]) -> str:
        """
        Convert SQL/FHIR records into text format for extraction
        
        Args:
            records: Dictionary with keys like 'medications', 'conditions', etc.
            
        Returns:
            Formatted text context
        """
        lines = []
        
        # Format medications
        if 'medications' in records and records['medications']:
            lines.append("=== MEDICATIONS ===")
            for med in records['medications']:
                med_text = f"- {med.get('description', 'Unknown medication')}"
                if 'start_date' in med:
                    med_text += f" (started: {med['start_date']})"
                if 'reason_description' in med:
                    med_text += f" for {med['reason_description']}"
                lines.append(med_text)
            lines.append("")
        
        # Format conditions
        if 'conditions' in records and records['conditions']:
            lines.append("=== CONDITIONS ===")
            for cond in records['conditions']:
                cond_text = f"- {cond.get('description', 'Unknown condition')}"
                if 'start_date' in cond:
                    cond_text += f" (diagnosed: {cond['start_date']})"
                lines.append(cond_text)
            lines.append("")
        
        # Format procedures
        if 'procedures' in records and records['procedures']:
            lines.append("=== PROCEDURES ===")
            for proc in records['procedures']:
                proc_text = f"- {proc.get('description', 'Unknown procedure')}"
                if 'date' in proc:
                    proc_text += f" (performed: {proc['date']})"
                if 'reason_description' in proc:
                    proc_text += f" for {proc['reason_description']}"
                lines.append(proc_text)
            lines.append("")
        
        # Format observations/lab results
        if 'observations' in records and records['observations']:
            lines.append("=== OBSERVATIONS/LAB RESULTS ===")
            for obs in records['observations']:
                obs_text = f"- {obs.get('description', 'Unknown observation')}"
                if 'value' in obs and 'units' in obs:
                    obs_text += f": {obs['value']} {obs['units']}"
                if 'date' in obs:
                    obs_text += f" ({obs['date']})"
                lines.append(obs_text)
            lines.append("")
        
        # Format immunizations
        if 'immunizations' in records and records['immunizations']:
            lines.append("=== IMMUNIZATIONS ===")
            for imm in records['immunizations']:
                imm_text = f"- {imm.get('display', 'Unknown vaccine')}"
                if 'date' in imm:
                    imm_text += f" (administered: {imm['date']})"
                lines.append(imm_text)
            lines.append("")
        
        return "\n".join(lines)
    
    def _call_llm_extraction(self, context: str) -> List[Dict[str, str]]:
        """
        Call LLM to extract triples from medical context
        
        Args:
            context: Formatted medical text
            
        Returns:
            List of extracted triples
        """
        if not self.client:
            logger.warning("No OpenAI client configured, returning empty triples")
            return []
        
        try:
            # Format the prompt
            user_prompt = self.EXTRACTION_USER_PROMPT.format(medical_context=context)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic for extraction
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Extract triples
            if 'triples' in parsed:
                return parsed['triples']
            else:
                logger.warning("Response missing 'triples' key")
                return []
                
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return []
    
    def _normalize_triples(self, triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize and validate extracted triples
        
        Args:
            triples: Raw triples from LLM
            
        Returns:
            Validated and normalized triples
        """
        normalized = []
        seen = set()  # For deduplication
        
        for triple in triples:
            # Validate structure
            if not all(k in triple for k in ['subject', 'predicate', 'object']):
                continue
            
            # Extract and normalize
            subject = str(triple['subject']).strip().lower()
            predicate = str(triple['predicate']).strip().lower()
            obj = str(triple['object']).strip().lower()
            
            # Skip if any part is empty
            if not all([subject, predicate, obj]):
                continue
            
            # Deduplicate
            triple_key = (subject, predicate, obj)
            if triple_key in seen:
                continue
            seen.add(triple_key)
            
            # Add normalized triple
            normalized.append({
                'subject': subject,
                'predicate': predicate,
                'object': obj
            })
        
        return normalized


# Example usage
if __name__ == "__main__":
    # Sample medical records
    sample_records = {
        'medications': [
            {
                'description': 'Lisinopril 10 MG Oral Tablet',
                'start_date': '2023-01-15',
                'reason_description': 'Hypertension'
            },
            {
                'description': 'Metformin 500 MG Oral Tablet',
                'start_date': '2023-03-20',
                'reason_description': 'Type 2 Diabetes'
            }
        ],
        'conditions': [
            {
                'description': 'Essential hypertension',
                'start_date': '2023-01-10'
            },
            {
                'description': 'Type 2 diabetes mellitus',
                'start_date': '2023-03-15'
            }
        ],
        'procedures': [
            {
                'description': 'Annual physical examination',
                'date': '2024-01-05',
                'reason_description': 'Routine checkup'
            }
        ]
    }
    
    # Initialize extractor (without client for demo)
    extractor = MedicalKGExtractor()
    
    # Format context
    context = extractor._format_medical_context(sample_records)
    print("=== Formatted Medical Context ===")
    print(context)
    print("\n" + "="*50 + "\n")
    
    # In real use with OpenAI client:
    # from openai import OpenAI
    # client = OpenAI(api_key="your-key")
    # extractor = MedicalKGExtractor(openai_client=client)
    # triples = extractor.extract_from_records(sample_records)
    # print(f"Extracted {len(triples)} triples")
