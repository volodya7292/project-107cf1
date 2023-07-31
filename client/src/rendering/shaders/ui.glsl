struct Rect {
    vec2 min;
    vec2 max;
};


bool isOutsideCropRegion(vec2 normCoord, Rect clipRect, float cornerRadius, vec2 logicalFrameSize) {
    vec2 cornerRadiusNorm = cornerRadius / logicalFrameSize;

    vec2 center = 0.5 * (clipRect.min + clipRect.max);
    vec2 dist = abs(normCoord - center);

    vec2 clipRectSize = clipRect.max - clipRect.min;
    vec2 unitDist = max(vec2(0.0), dist - (0.5 * clipRectSize - cornerRadiusNorm));
    float sd = pow(unitDist.x / max(0.0001, cornerRadiusNorm.x), 2) + pow(unitDist.y / max(0.0001, cornerRadiusNorm.y), 2);

    return sd > 1.0;
}
