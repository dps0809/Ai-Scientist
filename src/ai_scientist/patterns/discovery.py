from ai_scientist.models import DatasetProfile, Pattern


def discover_patterns(profile: DatasetProfile) -> tuple[Pattern, ...]:
    direction = "positive" if profile.feature_target_correlation >= 0 else "negative"
    return (
        Pattern(
            description=f"The feature has a {direction} relationship with the target.",
            evidence=f"Pearson correlation={profile.feature_target_correlation:.3f}",
            confidence=min(abs(profile.feature_target_correlation), 1.0),
        ),
    )
