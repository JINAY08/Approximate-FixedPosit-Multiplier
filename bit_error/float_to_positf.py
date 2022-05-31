import numpy as np

class fposit():
    def __init__(self,length,regime,es):
        self.length = length
        self.es = es 
        self.regime = regime 
    
    def extract(self,operand):
        total_length = self.length
        es = self.es
        regime_len = self.regime
        frac_length = total_length - es - regime_len - 1
        if(frac_length<0):
            return (None)
        k_max = regime_len-1
        k_min = -1*regime_len
        e_max = 2**es - 1
        s=0
        e=0
        k=0
        f=1
        if(operand==0):
            return(s,k,e,f-1)
        elif(operand<0):
            s=1
        operand = abs(operand)
        exp = int(np.floor(np.log2(operand)))
        useed= 2**es
        k=int(np.floor(exp/useed))
        e=exp-k*useed
        if(k>k_max):
            k=k_max
        elif(k<k_min):
            k=k_min
        
        if(e>e_max):
            e=e_max
        f=float(operand)/(2**exp)
        f_new=f-1
        for i in range(1,frac_length+1):
            f_new-=2**(-i)
            if(f_new<0):
                f_new+=2**(-i)
        # Rounding error begin
        rounding=2**(-1*(frac_length+1))
        # print(rounding)
        if(f_new>=rounding):
            f = f-f_new+2**(-1*(frac_length))
            if(f>=2):
                f=f/2
                e+=1
            if(e>2**es - 1):
                k+=1
                e=0

        else:
            f = f - f_new
        # Rounding error end
        return (s,k,e,f)




    def neg(self,string):
        out = ""
        # print(len(string))
        for i in range(len(string)):
            if(string[i]=="1"):
                out+="0"
            else:
                out+="1"
        # print(out)
        return out



    def float2posit(self,dec):
        total_length = self.length
        es = self.es
        regime_len = self.regime
        frac_length = total_length - es - regime_len - 1
        k_max = regime_len-1
        k_min = -1*regime_len
        e_max = 2**es - 1

        sign,k,e,frac = self.extract(dec)

        # print(sign,k,e,frac)
        if(sign==0 and k==0 and e==0 and frac == 0):
            return ("0"*total_length) # if the input is 0
        out=''

        if (k>=0):
            if(k==k_max):
                out_regime = '1'*(regime_len)
            elif(k<k_max):
                out_regime = '1'*(k+1) + "0"*(regime_len-k-1)

        else:
            if(k==k_min):
                out_regime = '0'*(regime_len)
            elif(k>k_min):
                out_regime = '0'*abs(k) + "1"*(regime_len-abs(k))

        out += out_regime

        e_bin = bin(e)[2:]

        if(len(e_bin)<es):
            e_bin = "0"*(es-len(e_bin)) + e_bin
        # print(e_bin)
        out+=e_bin

        frac -= 1
        frac_bin = ""
        for i in range(frac_length):
            # print(frac)
            frac*=2
            if(frac>=1):
                frac_bin +="1"
                frac -=1
            else:
                frac_bin +="0" 
        out+=frac_bin
        # print(out)
        # do it at last
        if(sign==0):
            out ='0' + out
        else:
            out ='1' + self.neg(out)
        return (out)
    



    def posit2float(self,posit_dec):

        total_length = self.length
        es = self.es
        regime_len = self.regime
        frac_length = total_length - es - regime_len - 1

        if(posit_dec=="0"*total_length):
            return (0.0)
        posit_in = posit_dec[0]
        if(posit_dec[0]=="1"):
            posit_in += self.neg(posit_dec[1:])
        else:
            posit_in += posit_dec[1:]

        # print(posit_in)
        # calculate k
        if(posit_in[1]=="1"):
            regime_bits = posit_in[1:regime_len+1]
        else:
            regime_bits = self.neg(posit_in[1:regime_len+1])
        count_k = 0
        # print(regime_bits)
        for i in range(regime_len):
            if(regime_bits[i]=="0"):
                break
            else:
                count_k+=1
        if(posit_in[1]=="1"):
            k = count_k - 1
        else:
            k = -1 * count_k
        # calculate es bits
        es_bits = posit_in[regime_len+1:regime_len+es+1]
        # print(es_bits)
        e = 0
        # print(es_bits)
        for i in range(es):
            if(es_bits[i]=="1"):
                # print(len(es_bits),i)
                e+= 2**(len(es_bits)-i-1)
        # print(e)
        # calculating total exp
        exp = (2**es)*k + e
        # print(exp)

        frac_bits = posit_in[1 + regime_len + es:]
        # print(frac_bits)
        frac = 1
        for i in range(len(frac_bits)):
            if(frac_bits[i]=="1"):
                frac += 2**(-(i+1)) 
        # print(frac)
        out = 2**(exp) * frac
        if(posit_in[0]=="0"):
            return (out)
        else:
            return (-1*out)
    



    def posit_error(self,dec,bit_pos):

        out = self.float2posit(dec)
        
        out_list = list(out)
        
        if(out_list[bit_pos] == 0):
            out[bit_pos] = '1'
        else:
            out_list[bit_pos] = '0'
        
        out = ''.join(out_list)
        # print(out)
        float_error = self.posit2float(out)

        return float_error






# posit = fposit(6,2,2)
# a=-1.48
# print(posit.extract(a))
# print(posit.float2posit(a))
# print(posit.posit2float(posit.float2posit(a)))
# print(posit.posit_error(a,2))