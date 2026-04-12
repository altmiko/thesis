import torch
from validator import TCP, UDP, ARP, ICMP, HTTP, HTTPS, SSH, SMTP, IRC, DNS, DHCP
from validator import FIN_FLAG, SYN_FLAG, RST_FLAG, PSH_FLAG, ACK_FLAG, ECE_FLAG, CWR_FLAG
from validator import ACK_CNT, SYN_CNT, FIN_CNT, RST_CNT, F_MIN, F_MAX, AVG, CONTINUOUS_IDX, BINARY_IDX

def compute_protocol_loss(x_adv, scaler_mean, scaler_scale, reduce=True):
    # x_adv: (B, 37)
    binary = x_adv[:, BINARY_IDX]
    cont = x_adv[:, CONTINUOUS_IDX]
    loss = torch.zeros(x_adv.size(0), device=x_adv.device)
    
    # 1. TCP/UDP Exclusive: TCP * UDP should be 0
    loss += torch.relu(binary[:, TCP] + binary[:, UDP] - 1.0)
    
    # 2. ARP -> not TCP, UDP, ICMP
    arp = binary[:, ARP]
    loss += (arp * binary[:, TCP])
    loss += (arp * binary[:, UDP])
    loss += (arp * binary[:, ICMP])
    
    # 3. ICMP -> not TCP, UDP
    icmp = binary[:, ICMP]
    loss += (icmp * binary[:, TCP])
    loss += (icmp * binary[:, UDP])
    
    # 4. App -> TCP
    app_tcp = binary[:, HTTP] + binary[:, HTTPS] + binary[:, SSH] + binary[:, SMTP] + binary[:, IRC]
    # For every app_tcp active, TCP must be 1 => tcp_missing = relu(1.0 - tcp)
    loss += (app_tcp * torch.relu(1.0 - binary[:, TCP]))
    
    # 5. DNS -> TCP or UDP
    dns = binary[:, DNS]
    transport = torch.clamp(binary[:, TCP] + binary[:, UDP], max=1.0)
    loss += (dns * torch.relu(1.0 - transport))
    
    # 6. DHCP -> UDP
    dhcp = binary[:, DHCP]
    loss += (dhcp * torch.relu(1.0 - binary[:, UDP]))
    
    # 7. TCP flags -> TCP
    flags_active = cont[:, FIN_FLAG] + cont[:, SYN_FLAG] + cont[:, RST_FLAG] + cont[:, PSH_FLAG] + cont[:, ACK_FLAG] + cont[:, ECE_FLAG] + cont[:, CWR_FLAG]
    flags_active_bool = torch.sigmoid(flags_active * 10) # smooth approx of >0
    loss += (flags_active_bool * torch.relu(1.0 - binary[:, TCP]))
    
    # 8. Flag -> count consistency
    pairs = [(FIN_FLAG, FIN_CNT), (SYN_FLAG, SYN_CNT), (ACK_FLAG, ACK_CNT), (RST_FLAG, RST_CNT)]
    for f_i, c_i in pairs:
        f_active = torch.sigmoid(cont[:, f_i] * 10)
        c_missing = torch.relu(1e-4 - cont[:, c_i]) # if count <= 0
        loss += (f_active * c_missing)
        
    # 9. Min <= AVG <= Max
    # Unscale: raw = scaled * scale + mean
    raw_min = cont[:, F_MIN] * scaler_scale[F_MIN] + scaler_mean[F_MIN]
    raw_max = cont[:, F_MAX] * scaler_scale[F_MAX] + scaler_mean[F_MAX]
    raw_avg = cont[:, AVG] * scaler_scale[AVG] + scaler_mean[AVG]
    
    loss += torch.relu(raw_min - raw_avg)
    loss += torch.relu(raw_avg - raw_max)
    
    # 10. Non-neg counts (in scaled space < -3)
    loss += torch.relu(-3.0 - cont[:, ACK_CNT])
    loss += torch.relu(-3.0 - cont[:, SYN_CNT])
    loss += torch.relu(-3.0 - cont[:, FIN_CNT])
    loss += torch.relu(-3.0 - cont[:, RST_CNT])

    if reduce:
        return loss.sum()
    return loss
