"""Conservative, sentence-level evidence checks for Experiment 2 prose.

The grammar covers explicit direction, rank, influence, reported probability,
and simple observations. It flags recognizable causal and scale language.
Everything outside the grammar is recorded as ambiguous and requires review.
"""

from dataclasses import dataclass
import math
import re

from llm_verification import contribution_direction, expected_drivers, reference_rows, sentences


# Exact keys and the labels supplied with the case are also approved aliases.
ALIASES = {
    "age": ("patient age", "patient's age"),
    "prior_admissions": ("prior hospital admissions", "previous admissions"),
    "creatinine": ("serum creatinine", "creatinine level"),
    "sodium": ("serum sodium", "sodium level"),
}
RISK = r"(?:the )?(?:model(?:'s)? )?(?:predicted |readmission )*(?:risk|probability|prediction)"
RELATIVE = r"(?: relative to (?:the )?(?:model(?:'s)? )?(?:reference(?: prediction)?|baseline))?"
DIRECTION = re.compile(
    rf"(?P<subject>.+?) (?:(?P<verb>increases?|decreases?|raises?|lowers?|reduces?) {RISK}"
    rf"|(?:is|are) associated with (?:a )?(?P<association>higher|lower) {RISK}){RELATIVE}", re.I,
)
RANK = re.compile(
    r"(?P<subject>.+?) (?:is|are) (?:the |a )?(?P<rank>strongest|largest|top three|top 3|"
    r"second strongest|third strongest|second-largest|third-largest) (?:model )?"
    r"(?:contributor|driver|factor)s?(?: (?:to|of) (?:the )?(?:model's )?prediction)?", re.I,
)
INFLUENCE = re.compile(
    rf"(?P<subject>.+?) (?:(?:influences?|contributes? to|affects?) {RISK}"
    r"|(?:is|are) (?:a |an )?(?:meaningful |important )?(?:model )?(?:driver|contributor|factor)s?)", re.I,
)
PREDICTION = re.compile(
    r"(?:the )?model (?:estimates|predicts) (?:a |an )?(?P<percent>\d+(?:\.\d+)?)%"
    r" (?:30[- ]day )?readmission (?:probability|risk)", re.I,
)
CAUSAL = re.compile(r"\b(?:causes?|caused|causing|leads? to|led to|results? in|"
                    r"prevents?|prevented|causal(?:ly)?|due to)\b", re.I)
SCALE = re.compile(r"\b(?:percentage|percent|probability)[ -]points?\b|"
                   r"\b(?:shap|contribut\w*)\b.*?(?:\d\s*%|\bpercent\b)", re.I)
SCALE_DESCRIPTION = re.compile(
    r"(?:shap (?:values|contributions)|the contributions) (?:are|describe) "
    r"(?:changes in )?(?:the model's )?(?:raw )?log[- ]odds"
    r"(?:,? not (?:percentage|probability)[ -]point changes)?"
    r"|(?:shap (?:values|contributions)|the contributions) are not "
    r"(?:percentage|probability)[ -]point changes", re.I,
)
CLINICAL = re.compile(
    r"\b(?:diagnos\w*|kidney (?:failure|disease|injury)|renal (?:failure|impairment|disease)|"
    r"hyponatremia|hypernatremia|anemia|treatment|medication|should receive|should be treated)\b", re.I,
)
ABSTENTION = re.compile(
    r"(?:the model contributions are unknown|the drivers are uncertain|"
    r"i cannot (?:reliably )?(?:identify|determine) (?:the )?(?:model(?:'s)? )?(?:drivers|contributions)|"
    r"no supplied feature has a non-negligible SHAP contribution)", re.I,
)
ABSENCE = r"(?:is|was) (?:missing|unavailable|not measured|not recorded)"
MEASUREMENT = r"(?:is|was|measures|measured|of|at|=|:)\s*(?P<value>-?\d+(?:\.\d+)?)(?P<unit>.*)"


@dataclass(frozen=True)
class Claim:
    text: str
    status: str
    code: str
    detail: str
    feature: str | None = None


@dataclass(frozen=True)
class NarrativeResult:
    claims: tuple[Claim, ...]
    passed: bool


def _aliases(case):
    aliases = {}
    for feature in case.features:
        name = feature["name"]
        for alias in (name, name.replace("_", " "), feature["label"], *ALIASES.get(name, ())):
            if alias:
                aliases.setdefault(alias.casefold(), set()).add(name)
    return {alias: next(iter(names)) for alias, names in aliases.items() if len(names) == 1}


def verify_narrative(case, answer) -> NarrativeResult:
    features = {row["name"]: row for row in case.features}
    aliases = _aliases(case)
    names = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    mention = re.compile(rf"(?<!\w)(?:{names})(?!\w)", re.I) if names else re.compile(r"(?!)")
    clause_split = re.compile(r";\s*|,?\s+while\s+|,?\s+and\s+(?=" + mention.pattern + ")", re.I)
    rows = reference_rows(case)
    values = {row["feature"]: row["shap"] for row in rows}
    top = tuple(driver.feature for driver in expected_drivers(case))
    observed = {driver.feature: driver.direction for driver in answer.drivers}
    claims = []

    for sentence in sentences(answer.explanation):
        clean = sentence.replace("’", "'").replace("`", "").strip().rstrip(".!?")
        for clause in clause_split.split(clean):
            claims.extend(_check_clause(clause.strip(), case, features, aliases, mention,
                                        values, top, observed))
    return NarrativeResult(
        tuple(claims), all(claim.status in ("supported", "non_attribution") for claim in claims)
    )


def _check_clause(text, case, features, aliases, mention, values, top, observed):
    def claim(status, code, detail, feature=None):
        return Claim(text, status, code, detail, feature)

    if SCALE_DESCRIPTION.fullmatch(text):
        return [claim("non_attribution", "scale_description", "The description matches the SHAP output scale.")]

    hazards = [
        claim("contradicted", code, detail)
        for pattern, code, detail in (
            (CAUSAL, "causal", "SHAP does not establish causation."),
            (SCALE, "scale", "SHAP contributions are raw log-odds, not probability-point changes."),
            (CLINICAL, "clinical", "The supplied prediction and feature attributions do not support clinical conclusions."),
        )
        if pattern.search(text)
    ]
    # An observed-level assertion is invalid even when embedded in a longer clause.
    for match in mention.finditer(text):
        name = aliases[match[0].casefold()]
        if features[name]["value"] is not None:
            continue
        prefix = text[:match.start()].rstrip()
        suffix = text[match.end():].lstrip()
        if re.search(r"\b(?:low|high|normal|elevated|reduced)$", prefix, re.I) or re.match(
            r"(?:(?:level|measurement)\s+)?(?:(?:is|was|of|at|=|:)\s*|(?:measures?|measured)\s+)"
            r"(?:-?\d|(?:low|high|normal|elevated|reduced|observed|recorded|measured)\b)",
            suffix, flags=re.I,
        ):
            hazards.append(claim("contradicted", "missing_observation", f"{name} was unavailable.", name))
    if hazards:
        return hazards

    for pattern in (DIRECTION, RANK, INFLUENCE):
        match = pattern.fullmatch(text)
        if not match:
            continue
        subject = match["subject"]
        name = aliases.get(subject.casefold()) or aliases.get(subject.casefold().removeprefix("the "))
        if name is None:
            if mention.search(subject):
                return [claim("ambiguous", "ambiguous_subject", "The feature claim has an unrecognized qualifier.")]
            return [claim("contradicted", "unsupported_feature", f"{subject} is not a supplied model feature.")]
        sign = contribution_direction(values[name])
        if sign is None:
            return [claim("contradicted", "negligible", f"{name} has a negligible contribution.", name)]
        if pattern is DIRECTION:
            word = (match["verb"] or match["association"]).lower()
            stated = "positive" if word.startswith(("increase", "raise", "higher")) else "negative"
            errors = []
            if stated != sign:
                errors.append(claim("contradicted", "direction", f"{name} has a {sign} contribution.", name))
            if name in observed and stated != observed[name]:
                errors.append(claim("contradicted", "driver_conflict", f"Prose disagrees with the JSON direction for {name}.", name))
            return errors or [claim("supported", "direction", f"{name} has a {sign} contribution.", name)]
        if pattern is RANK:
            rank = match["rank"].lower()
            if rank in ("top three", "top 3"):
                valid = name in top
            else:
                position = 1 if rank.startswith("second") else 2 if rank.startswith("third") else 0
                valid = position < len(top) and abs(values[name]) == abs(values[top[position]])
            return [claim("supported" if valid else "contradicted", "ranking",
                          f"Reference order: {', '.join(top)}.", name)]
        return [claim("supported", "influence", f"{name} has a non-negligible contribution.", name)]

    if match := PREDICTION.fullmatch(text):
        displayed = match["percent"]
        decimals = len(displayed.partition(".")[2])
        valid = math.isclose(float(displayed), 100 * case.predicted_risk,
                             rel_tol=0, abs_tol=0.5 * 10 ** -decimals + 1e-12)
        return [claim("non_attribution" if valid else "contradicted", "prediction",
                      f"Supplied predicted probability: {case.predicted_risk:.2%}.")]
    if ABSTENTION.fullmatch(text):
        if text.lower().startswith("no supplied") and top:
            return [claim("contradicted", "false_abstention", "The reference contains non-negligible contributions.")]
        return [claim("non_attribution", "abstention", "No directional or ranking claim.")]
    if match := mention.match(text):
        name = aliases[match[0].casefold()]
        rest = text[match.end():].strip()
        feature = features[name]
        if re.fullmatch(ABSENCE, rest, flags=re.I):
            valid = feature["value"] is None
        elif measurement := re.fullmatch(MEASUREMENT, rest, flags=re.I):
            unit = measurement["unit"].strip().casefold()
            valid = (isinstance(feature["value"], (int, float))
                     and feature["value"] == float(measurement["value"])
                     and (not unit or unit == (feature["unit"] or "").casefold()))
        else:
            valid = None
        if valid is not None:
            return [claim("non_attribution" if valid else "contradicted", "observation",
                          f"Checked the reported observation against the supplied value for {name}.", name)]
    return [claim("ambiguous", "unparsed", "The verifier cannot determine the faithfulness of this clause.")]
