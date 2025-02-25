import math


###### FUNCTIONS FOR DESCRIPTIVE STATISTICS ######

def ft_count(data):
    """Calculate the number of non-null observations."""
    return sum(1 for x in data if x is not None and not math.isnan(x))

def ft_mean(data):
    """Calculate the arithmetic mean of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else sum(clean_data) / len(clean_data)

def ft_std(data):
    """Calculate the standard deviation of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 2:
        return float('nan')
    mean = ft_mean(clean_data)
    return math.sqrt(sum((x - mean) ** 2 for x in clean_data) / (len(clean_data) - 1))

def ft_min(data):
    """Find the minimum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else min(clean_data)

def ft_max(data):
    """Find the maximum value in the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    return float('nan') if not clean_data else max(clean_data)

def ft_percentile(data, q):
    """Calculate the qth percentile of the data."""
    clean_data = sorted([x for x in data if x is not None and not math.isnan(x)])
    if not clean_data:
        return float('nan')
    if len(clean_data) == 1:
        return clean_data[0]
    position = (len(clean_data) - 1) * q
    floor, ceil = math.floor(position), math.ceil(position)
    if floor == ceil:
        return clean_data[int(position)]
    d0 = clean_data[floor] * (ceil - position)
    d1 = clean_data[ceil] * (position - floor)
    return d0 + d1

def ft_median(data):
    """Calculate the median (50th percentile) of the data."""
    return ft_percentile(data, 0.5)

def ft_iqr(data):
    """Calculate the Interquartile Range (IQR) of the data."""
    q75, q25 = ft_percentile(data, 0.75), ft_percentile(data, 0.25)
    return float('nan') if math.isnan(q75) or math.isnan(q25) else q75 - q25

def ft_skewness(data):
    """Calculate the skewness of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 3:
        return float('nan')
    mean, std = ft_mean(clean_data), ft_std(clean_data)
    if std == 0:
        return float('nan')
    m3 = sum((x - mean) ** 3 for x in clean_data) / len(clean_data)
    return m3 / (std ** 3)

def ft_kurtosis(data):
    """Calculate the kurtosis of the data."""
    clean_data = [x for x in data if x is not None and not math.isnan(x)]
    if len(clean_data) < 4:
        return float('nan')
    mean, std = ft_mean(clean_data), ft_std(clean_data)
    if std == 0:
        return float('nan')
    m4 = sum((x - mean) ** 4 for x in clean_data) / len(clean_data)
    return (m4 / (std ** 4)) - 3

def ft_cv(data):
    """Calculate the Coefficient of Variation (CV) of the data."""
    mean, std = ft_mean(data), ft_std(data)
    return float('nan') if mean == 0 or math.isnan(mean) or math.isnan(std) else abs(std / mean)



