def Ratio(numerator, denominator):
    '''
    computes thr ratio of poi messages sent or from
    Where parameter poi_message is total msgs sent to or from poi
    and parameter to_from_msgs is total sent or recieved messages
    
    '''
    
    from decimal import Decimal
    if numerator == 'NaN':
        return float(0)  
    if denominator == 'NaN':
        return float(0)
    else:
        ratio = float(numerator) / float((denominator))
        return ratio

def Inv(val):
    global inverse
    if val != 'NaN':
        inverse = float(1)/float(val)
        return inverse
    else:
        return inverse