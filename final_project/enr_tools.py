def poi_msgratio(poi_messages,to_from_msgs):
    '''
    computes thr ratio of poi messages sent or from
    Where parameter poi_message is total msgs sent to or from poi
    and parameter to_from_msgs is total sent or recieved messages
    
    '''
    
    from decimal import Decimal
    if poi_messages != 'NaN' or  to_from_msgs != 'NaN':
        return 0
    else:
        poi_ratio = Decimal(poi_messages) / Decimal(to_from_msgs)
        return poi_ratio


    
def creat_msgf(num, denum, feat_name):
    '''
    return dictionary of with key value pairs of feature and value
    '''
    feat ={feat_name: poi_msgratio(num, denum)}

    return feat
print poi_msgratio(142,2134)
