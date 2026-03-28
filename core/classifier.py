def classify(do, ph, sulphur, temp, turbidity):
    score = 0

    if do > 6: score += 1
    if 6.5 <= ph <= 8.5: score += 1
    if sulphur < 3: score += 1
    if temp < 28: score += 1
    if turbidity < 5: score += 1

    if score >= 4:
        return {"status": "Safe", "color": "green"}
    elif score >= 2:
        return {"status": "Moderate", "color": "orange"}
    else:
        return {"status": "Poor", "color": "red"}