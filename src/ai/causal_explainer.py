"""
AI Causal Explainer Module
==========================

This module integrates with the Google Gemini API to generate
human-readable business insights and causal explanations based
on the probabilistic outputs of the Bayesian forecasting model.

Responsibilities
----------------
- Format probabilistic data (Mean, HDI) into structured LLM prompts.
- Execute API calls to the Gemini model using exponential backoff.
- Return structured business insights explaining forecast deviations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import requests

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass(slots=True)
class CausalExplainer:
    """
    Generates causal business explanations using the Gemini API.

    Parameters
    ----------
    api_key : str
        The Google Gemini API key. If not provided, it attempts to
        load from the GEMINI_API_KEY environment variable.
    model_version : str
        The specific Gemini model to use for generation.
    """

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_version: str = "gemini-2.5-flash-preview-09-2025"

    def __post_init__(self) -> None:
        """Validate the presence of an API key."""
        if not self.api_key:
            logger.warning(
                "No API key provided. Causal explainer will fail if executed "
                "outside the integrated canvas environment."
            )

    def _call_gemini_api(self, prompt: str, system_instruction: str) -> Optional[str]:
        """
        Executes a POST request to the Gemini API with exponential backoff.
        """
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_version}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
        }
        headers = {"Content-Type": "application/json"}

        # Exponential Backoff Retry Logic (1s, 2s, 4s, 8s, 16s)
        delays = [1, 2, 4, 8, 16]
        for attempt, delay in enumerate(delays, start=1):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Extract text payload
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
                if text:
                    return text.strip()
                else:
                    logger.error("API Response missing text payload.")
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt == len(delays):
                    logger.error(f"Gemini API request failed after {len(delays)} attempts: {e}")
                    return f"Error: Unable to generate insight due to API failure ({e})."
                time.sleep(delay)

        return None

    def explain_anomaly(
        self,
        hierarchy_level: str,
        entity_name: str,
        actual_revenue: float,
        predicted_mean: float,
        hdi_lower: float,
        hdi_upper: float,
        spend: float,
    ) -> str:
        """
        Generates an explanation for a specific entity where actual performance
        deviated from the model's predicted HDI bounds.

        Parameters
        ----------
        hierarchy_level : str
            e.g., "Channel", "Campaign Type", "Campaign"
        entity_name : str
            e.g., "Google Ads", "Performance Max", "Pmax_Campaign_03"
        actual_revenue : float
            The observed revenue.
        predicted_mean : float
            The model's expected mean revenue.
        hdi_lower, hdi_upper : float
            The 94% HDI bounds.
        spend : float
            The actual spend for this entity.

        Returns
        -------
        str
            The AI-generated business insight.
        """
        if actual_revenue == 0 and spend > 0:
            status = "Complete Zero-Conversion Failure"
        elif actual_revenue < hdi_lower:
            status = "Underperformance (Below 94% Confidence Interval)"
        elif actual_revenue > hdi_upper:
            status = "Overperformance (Exceeded 94% Confidence Interval)"
        else:
            status = "Within Expected Bounds"
            # If it's normal, we don't necessarily need a causal deep dive,
            # but we can provide a summary.

        system_instruction = (
            "You are a Senior Digital Marketing Strategist and Data Scientist. "
            "Your job is to analyze Bayesian forecasting outputs and explain deviations "
            "to agency stakeholders. Be concise, causal, and operational. Do not use "
            "generic fluff. Mention 'saturation', 'adstock', or 'hierarchical effects' "
            "if relevant. Keep the response to 3-4 sentences."
        )

        prompt = f"""
        Analyze the following e-commerce marketing anomaly:
        
        Entity Level: {hierarchy_level}
        Entity Name: {entity_name}
        Status: {status}
        
        Data Points:
        - Actual Spend: ${spend:.2f}
        - Actual Revenue: ${actual_revenue:.2f}
        - Model Predicted Mean: ${predicted_mean:.2f}
        - Model 94% HDI Range: [${hdi_lower:.2f}, ${hdi_upper:.2f}]
        
        Based on this data, provide a causal hypothesis for why this {hierarchy_level} 
        performed this way, and suggest one immediate operational action the agency should take.
        """

        logger.info(f"Requesting AI causal insight for {entity_name} ({status})...")
        insight = self._call_gemini_api(prompt, system_instruction)
        return insight or "Insight generation failed."


# ---------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------

__all__ = [
    "CausalExplainer",
]
