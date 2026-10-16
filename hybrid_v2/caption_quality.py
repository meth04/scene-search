"""Content-neutral caption instructions and conservative deterministic checks."""
from __future__ import annotations

import re

from .common import clean_text

CAPTION_PROMPT_VERSION = "qwen2b_literal_v2"
CAPTION_VALIDATION_VERSION = "literal_consistency_v2"
TEMPORAL_PROMPT_VERSION = "qwen2b_temporal_action_v3"
TEMPORAL_VALIDATION_VERSION = "temporal_consistency_v3"
CAPTION_PROMPT = """Describe only visually supported facts in this dashcam clip.
The image contains six chronological views of one approximately ten-second clip,
read left-to-right across the first row, then left-to-right across the second row.

Keep the description concise and literal. Preserve the sequence's chronology.
Do not infer a temporal action from a single ambiguous view. State merging,
lane changing, overtaking, braking or pedestrian crossing only when the sequence
supports the action. If an action or spatial relation is uncertain, omit it.
Do not infer a traffic-light color unless it is clearly visible. Do not read or
invent sign/text content unless it is legible. Do not infer intent, causality,
emotion, identity, speed or danger. Avoid "no X" claims unless absence is
semantically important and confidently observable.

Return JSON with the following keys. Empty values below show only field types;
they are not scene examples. Do not copy examples or invent values to fill keys.
Use empty strings/lists for facts that are unknown or not confidently supported.
{
  "description": "",
  "objects": [],
  "ego_action": "",
  "other_actions": [],
  "relations": [],
  "road_context": [],
  "traffic_control": [],
  "weather": "",
  "events": [],
  "queries": [{"text":"", "action":"", "relation":"", "context":""}]
}
Fill description with one or two factual English sentences. Structured fields
must agree with that description. Queries may contain zero to three concise
English searches supported by visible facts. Do not generate negative/absence
queries. Never mention tile labels, frame numbers, timestamps, image layout or
the contact sheet. Return JSON only.
"""

TEMPORAL_ACTION_PROMPT = """The images are chronological samples from one dashcam video clip.
Read left-to-right across each row, then continue on the next row. Compare the
frames over time; do not treat them as unrelated images.
Ego means the camera vehicle. The visible camera vehicle's hood is not another
road user. Focus on other nearby road users, including vehicles appearing at
the image edges. Do not assign an ego color from the hood.

Track each salient vehicle using appearance, approximate color, left/right
side, and ahead/behind position relative to ego. Determine whether its relative
position changes across multiple chronological views. Capture vehicle type,
color, side, start/end position and a supported short action. Perfect IDs are
not required. Omit unclear color (lighting, blur, distance); use conservative
categories: black, white, gray/silver, red, blue, other/uncertain.

Distinguish another vehicle overtaking ego from ego overtaking another vehicle.
Look for the same vehicle beside/behind ego earlier and ahead later, or an
equivalent visible sequence. A vehicle simply ahead is not proof of overtaking.
Inspect these event categories as a checklist, never assume they occurred:
overtaking; being overtaken; lane changing; merging; cutting in;
braking/decelerating; stopping; turning left; turning right; pedestrian crossing;
cyclist passing; motorcycle passing; vehicle entering from a side road;
vehicle exiting the road.
Inspect relations: left/right of ego, ahead/behind, adjacent/same lane, crossing
in front, moving from behind to beside, beside to ahead, or either side toward
ego's lane. Do not claim both same and adjacent lane for one vehicle at one time.

Include a clearly supported short event in description even if it occurs only
during part of the clip. Do not infer actions from one ambiguous frame. Omit
uncertain actions rather than guessing, but do not omit every short event.
Do not invent make/model, intent, speed, light colors or illegible sign text.
Avoid negative filler: omit unimportant absence claims instead of listing them.

Return JSON only. Description: one to three concise English sentences, starting
with the salient event if supported, then brief road context. Fields must agree
on actor, color, side, lane and motion. Empty values below are schema types only,
not scene examples. Lists must contain short English STRINGS, never dictionaries
or object IDs. Only queries may contain dictionaries with the shown four keys.
Use empty strings/lists for unknown fields; at most three positive queries.
Never mention frames, timestamps, labels, tiles, layout or the contact sheet.
{
  "description": "",
  "objects": [],
  "ego_action": "",
  "other_actions": [],
  "relations": [],
  "road_context": [],
  "traffic_control": [],
  "weather": "",
  "events": [],
  "queries": [{"text":"", "action":"", "relation":"", "context":""}]
}
"""


def prompt_for(mode="literal", frame_count=6):
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 2:
        raise ValueError("Caption sampling requires at least two chronological frames")
    if mode == "temporal":
        return TEMPORAL_ACTION_PROMPT, TEMPORAL_PROMPT_VERSION, TEMPORAL_VALIDATION_VERSION
    if mode == "literal":
        prompt = CAPTION_PROMPT if frame_count == 6 else CAPTION_PROMPT.replace("six chronological", f"{frame_count} chronological")
        return prompt, CAPTION_PROMPT_VERSION, CAPTION_VALIDATION_VERSION
    raise ValueError("Unknown caption mode")


ARTIFACT = re.compile(r"\bt\d+(?:/\d+)?\b|\b(?:frame|image|tile)\s*\d+\b|\b(?:first|last|final|initial|second|third|fourth|fifth|sixth|seventh|eighth)\s+(?:frames?|images?|tiles?)\b|\b(?:six|6|eight|8)\s+frames\b|contact[- ]sheet|\b(?:schema|prompt example|insert description|placeholder)\b", re.I)
LIST_FIELDS = ("objects", "other_actions", "relations", "road_context", "traffic_control", "events")


def _phrase_key(value):
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def _negative_only(sentence):
    # Drop whole negative sentences only. Mixed positive/negative sentences stay
    # intact: splitting or rewriting them could invent an affirmative assertion.
    return bool(re.match(r"^(?:there (?:is|are|were) no\b|no\b|neither\b|"
                         r"(?:the )?(?:ego )?vehicle (?:is|was) not\b)", sentence.strip(), re.I)
                and not re.search(r"\b(?:but|however|although)\b", sentence, re.I))


def checked_caption(obj, temporal=False):
    """Normalize the historical schema, then drop uncertain contradictory fields.

    This checks textual consistency only; it cannot certify visual truth.
    Artifact sentences are discarded, never rewritten as replacement facts.
    If no literal description remains, the caller preserves a fallback row.
    """
    from .caption_dataset import normalize
    schema_warnings = []
    if temporal and isinstance(obj, dict):
        obj = dict(obj)
        for field in LIST_FIELDS:
            values = obj.get(field, [])
            if isinstance(values, list):
                obj[field] = [value for value in values if isinstance(value, str)]
                if len(obj[field]) != len(values):
                    schema_warnings.append(f"{field}: dropped non-string schema value")
            elif not isinstance(values, str):
                obj[field] = []
                schema_warnings.append(f"{field}: dropped invalid list schema")
    result = normalize(obj)
    original_description = result["description"]
    warnings = list(obj.get("caption_warnings", [])) if isinstance(obj.get("caption_warnings", []), list) else []
    warnings.extend(schema_warnings)
    if ARTIFACT.search(result["description"]):
        sentences = re.split(r"(?<=[.!?])\s+", result["description"])
        sentences = [sentence for sentence in sentences if not ARTIFACT.search(sentence)]
        if not sentences:
            raise ValueError("Caption description contains only prompt-artifact text")
        result["description"] = " ".join(sentences)
        warnings.append("description: dropped prompt-artifact sentence")
    for field in LIST_FIELDS:
        values, seen = [], set()
        for value in result[field]:
            key = _phrase_key(value)
            if ARTIFACT.search(value):
                warnings.append(f"{field}: dropped prompt artifact")
            elif key in seen:
                warnings.append(f"{field}: dropped duplicate")
            else:
                seen.add(key)
                values.append(value)
        result[field] = values
    # Identical actor/action statements repeated across fields add no information.
    seen = set()
    for field in ("other_actions", "relations", "events"):
        values = []
        for value in result[field]:
            if _phrase_key(value) in seen:
                warnings.append(f"{field}: dropped repeated action/relation/event")
            else:
                seen.add(_phrase_key(value))
                values.append(value)
        result[field] = values
    for field in ("ego_action", "weather"):
        if ARTIFACT.search(result[field]):
            result[field] = ""
            warnings.append(f"{field}: dropped prompt artifact")
    desc = original_description.lower()
    no_lights = re.search(r"\b(?:no|without)\s+(?:visible\s+)?traffic\s+(?:lights?|signals?|signs?\s+or\s+signals?)\b", desc)
    no_signs = re.search(r"\b(?:no|without)\s+(?:visible\s+)?traffic\s+signs?\b", desc)
    no_lights = no_lights or re.search(r"\btraffic\s+(?:lights?|signals?)\s+(?:are|is)\s+not\s+visible\b", desc)
    colors = set(re.findall(r"\b(red|green|yellow|amber)\s+(?:traffic\s+)?(?:lights?|signals?)\b", desc))
    values = []
    for value in result["traffic_control"]:
        low = value.lower()
        is_light = bool(re.search(r"\b(?:lights?|signals?)\b", low))
        color = set(re.findall(r"\b(red|green|yellow|amber)\b", low))
        contradiction = ((no_lights and is_light) or (no_signs and "sign" in low)
                         or (is_light and len(colors) == 1 and color and not color.intersection(colors)))
        if contradiction:
            warnings.append("traffic_control: dropped claim contradicting description")
        else:
            values.append(value)
    result["traffic_control"] = values
    denied_actions = []
    for pattern, denied in ((r"\b(?:no|not|without)\b[^.!?;]{0,45}\b(?:overtaking|passing)\b", r"\b(?:overtaking|passing)\b"),
                            (r"\b(?:no|not|without)\b[^.!?;]{0,45}\b(?:changing lanes|lane changing)\b", r"\b(?:changing lanes|lane changing)\b"),
                            (r"\b(?:no|not|without)\b[^.!?;]{0,45}\bmerging\b", r"\bmerging\b")):
        if re.search(pattern, desc):
            denied_actions.append(denied)
    for field in ("other_actions", "relations", "events"):
        values = []
        for value in result[field]:
            if re.search(r"\b(?:no|not|without|possibly|maybe|uncertain)\b", value, re.I) or any(re.search(p, value, re.I) for p in denied_actions):
                warnings.append(f"{field}: dropped negative/uncertain/contradictory claim")
            else:
                values.append(value)
        result[field] = values
    queries, seen = [], set()
    for query in result["queries"]:
        if any(ARTIFACT.search(query[key]) for key in ("text", "action", "relation", "context")):
            warnings.append("queries: dropped prompt artifact")
        elif any(re.search(r"\b(?:no|without|not visible)\b", query[key], re.I) for key in ("text", "action", "relation", "context")):
            warnings.append("queries: dropped absence query")
        elif query["text"].casefold() in seen:
            warnings.append("queries: dropped duplicate")
        else:
            seen.add(query["text"].casefold())
            queries.append(query)
    result["queries"] = queries
    if temporal:
        sentences, seen = [], set()
        for sentence in re.split(r"(?<=[.!?])\s+", result["description"]):
            key = _phrase_key(sentence)
            if _negative_only(sentence):
                warnings.append("description: dropped negative-only filler sentence")
            elif key in seen:
                warnings.append("description: dropped duplicate sentence")
            else:
                seen.add(key)
                sentences.append(sentence)
        if not sentences:
            raise ValueError("Caption description contains only negative filler")
        result["description"] = " ".join(sentences)
        for field in LIST_FIELDS:
            kept = []
            for value in result[field]:
                if re.match(r"^(?:no|without|not|none)\b", value, re.I):
                    warnings.append(f"{field}: dropped negative filler")
                else:
                    kept.append(value)
            result[field] = kept
        if re.search(r"\b(?:no|not|without|possibly|maybe|uncertain)\b", result["ego_action"], re.I):
            result["ego_action"] = ""
            warnings.append("ego_action: dropped negative/uncertain action")
        # Normalize-generated query fallbacks may copy the removed filler.
        result["queries"] = [q for q in result["queries"] if not any(
            re.search(r"\b(?:no|not|without|neither)\b", q[k], re.I)
            for k in ("text", "action", "relation", "context"))]
    result["caption_warnings"] = list(dict.fromkeys(clean_text(w) for w in warnings))
    return result
