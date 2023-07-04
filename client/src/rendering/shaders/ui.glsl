struct Rect {
    vec2 min;
    vec2 max;
};

bool isOutsideCropRegion(vec2 normCoord, Rect clip_rect) {
    return any(lessThan(normCoord, clip_rect.min)) || any(greaterThan(normCoord, clip_rect.max));
}
