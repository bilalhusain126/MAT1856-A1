import numpy as np
from datetime import datetime
from scipy import optimize
from scipy import interpolate
import numpy_financial as npf
from dateutil.relativedelta import relativedelta

PREV_COUPON_DATE = '01-09-2024'  



class Bond:
   def __init__(self, price_arr, maturity_date, coupon):
       self.price_arr = price_arr
       self.maturity_date = maturity_date
       self.coupon = coupon



def compute_dirty_price(clean_price, coupon, current_date_str, prev_coupon_date_str=PREV_COUPON_DATE, notional=100):
    
    current_date = datetime.strptime(current_date_str, '%d-%m-%Y')
    prev_coupon_date = datetime.strptime(prev_coupon_date_str, '%d-%m-%Y') 
    
    accrued_interest = notional * (coupon/100) * ((current_date - prev_coupon_date).days / 365)
    dirty_price = accrued_interest + clean_price
    
    return dirty_price



def compute_YTM(bond, day_index, current_date_str, notional=100):
    
    current_date = datetime.strptime(current_date_str, '%d-%m-%Y')
    maturity_date = datetime.strptime(bond.maturity_date, '%d-%m-%Y') 
    
    time_to_maturity = (maturity_date - current_date).days / 365.0
    coupon_payment = (bond.coupon / 100) * notional / 2  # Semi-annual coupon payment
    price = compute_dirty_price(bond.price_arr[day_index], bond.coupon, current_date_str)
    
    # Generate cash flow array
    cash_flows = [-price]  

    periods = int(2*time_to_maturity)+1
    
    # Add coupon payments for each period
    for t in range(periods-1):
        cash_flows.append(coupon_payment)
        
    cash_flows.append(coupon_payment + notional)

    # Compute semi-annual YTM using IRR function
    ytm = npf.irr(cash_flows)  
    
    return ytm*200, time_to_maturity



def compute_spot_curve(bonds, day_index, current_date_str, notional=100):
    
    current_date = datetime.strptime(current_date_str, '%d-%m-%Y')
    spot_rates = {}
    
    for bond in bonds:
        maturity_date = datetime.strptime(bond.maturity_date, '%d-%m-%Y')
        time_to_maturity = (maturity_date - current_date).days / 365.0
        coupon_payment = (bond.coupon / 100) * notional / 2  

        price = compute_dirty_price(bond.price_arr[day_index], bond.coupon, current_date_str)
        #price = bond.price_arr[day_index]  
        
        if len(spot_rates) == 0:
            # Zero-coupon bond case 
            spot_rate = -np.log(price / (notional + coupon_payment)) / time_to_maturity
        else:
            # Solve for spot rate iteratively using bootstrapping
            def spot_rate_equation(r):
                total_p = sum(coupon_payment*np.exp(-spot_rates[t]*t) for t in spot_rates)
                total_p += (coupon_payment + notional) * np.exp(-r*time_to_maturity)
                return total_p - price
            
            spot_rate = optimize.newton(spot_rate_equation, 0.05)  # Initial guess of 5%
        
        spot_rates[time_to_maturity] = spot_rate  

    spot_rates = {t: r*100 for t, r in spot_rates.items()}
    return spot_rates



def compute_forward_rate(spot_rates, t, n):

    r_interp = interpolate.interp1d(list(spot_rates.keys()), list(spot_rates.values()), kind='linear', fill_value="extrapolate")
    forward_rates = {}

    S_t = r_interp(t)
    S_tn = r_interp(t+n)
        
    # Calculate the forward rate Ft,t+n using the semi-annual compounding formula
    forward_rate = ( ((1 + S_tn)**(2*(t+n))) / ((1 + S_t)**(2*t)) ) ** (1/(2*n) ) - 1
    
    return forward_rate
    