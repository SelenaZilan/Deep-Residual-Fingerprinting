import pandas as pd
import numpy as np
import os
import subprocess
import pickle
from scapy.all import rdpcap, IP, TCP, UDP
from datetime import timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

TIME_OFFSET_HOURS = 3      
MAX_PACKETS = 50           
PAYLOAD_LIMIT = 216        
SAVE_INTERVAL = 500  
NUM_WORKERS = 8

CSV_FILE = "CIC-IDS-2017/GeneratedLabelledFlows/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
PCAP_FILE = "CIC-IDS-2017/raw_data_pcap/Friday-WorkingHours.pcap"
CSV_NAME = os.path.basename(CSV_FILE)
CSV_NAME = CSV_NAME[:-len(".pcap_ISCX.csv")]
OUTPUT_DIR = f"preprocessed/{CSV_NAME}_maxpkts{MAX_PACKETS}_payload{PAYLOAD_LIMIT}" 

samples_per_label = 1000

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

def clean_and_truncate_packet(pkt, payload_limit=216):
    if IP not in pkt: return None
    
    # 1. IP header handling: fixed 20 bytes.
    # Keep the first 20 bytes and anonymize version, service type, protocol, and IP addresses.
    ip_header = bytearray(bytes(pkt[IP])[:20])
    ip_header[0] = 0; ip_header[1] = 0; ip_header[9] = 0
    ip_header[12:20] = b'\x00' * 8

    # 2. L4 header alignment: always pad to 20 bytes.
    # Preallocate a 20-byte zero-filled placeholder.
    l4_padded = bytearray(20) 
    
    if TCP in pkt:
        # TCP headers are represented with 20 bytes.
        l4_raw = bytearray(bytes(pkt[TCP])[:20])
        l4_raw[0:4] = b'\x00' * 4  # Anonymize source/destination ports.
        l4_padded[:] = l4_raw      # Fill all 20 bytes.
        payload = bytes(pkt[TCP].payload)
        
    elif UDP in pkt:
        # UDP headers only use 8 bytes.
        l4_raw = bytearray(bytes(pkt[UDP])[:8])
        l4_raw[0:4] = b'\x00' * 4  # Anonymize source/destination ports.
        l4_padded[:8] = l4_raw     # Fill the first 8 bytes; keep the remaining 12 bytes as zero.
        payload = bytes(pkt[UDP].payload)
        
    else:
        # Other protocols, such as ICMP, GRE, or ESP.
        # Keep l4_padded as a 20-byte zero placeholder so payload still starts at byte 40.
        payload = bytes(pkt[IP].payload)

    # 3. Final layout: 20 bytes IP + 20 bytes L4 + payload.
    # Convert the byte stream into a uint8 vector for the model.
    full_vector = ip_header + l4_padded + payload
    return np.frombuffer(full_vector[:payload_limit + 40], dtype=np.uint8)


def process_single_flow(pcap_path, src_ip):
    try:
        packets = rdpcap(pcap_path)
    except: return None
    first_fwd = next((i for i, p in enumerate(packets) if IP in p and p[IP].src == src_ip), -1)
    if first_fwd == -1: return None
    selected_packets = packets[first_fwd : first_fwd + MAX_PACKETS]
    flow_features, last_time = [], None
    for pkt in selected_packets:
        if IP not in pkt: continue
        main_feat = clean_and_truncate_packet(pkt, PAYLOAD_LIMIT)
        curr_time = float(pkt.time)
        iat = (curr_time - last_time) if last_time else 0.0
        last_time = curr_time
        flow_features.append({
            'byte_vector': main_feat, 'iat': iat,
            'direction': 1 if pkt[IP].src == src_ip else 0,
            'original_len': len(pkt), 'protocol': pkt[IP].proto
        })
    return flow_features

def worker_task(idx, flow):
    """
    Run the per-flow extraction pipeline: editcap -> tcpdump -> scapy.
    """
    raw_timestamp = str(flow['Timestamp'])
    dt = pd.to_datetime(raw_timestamp, dayfirst=True)
    if "Afternoon" in CSV_FILE and dt.hour < 9: dt = dt.replace(hour=dt.hour + 12)
    dt_utc = dt + timedelta(hours=TIME_OFFSET_HOURS)
    duration = float(flow['Flow Duration']) / 1_000_000
    
    start_w = (dt_utc - timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
    end_w = (dt_utc + timedelta(minutes=1) + timedelta(seconds=duration)).strftime('%Y-%m-%d %H:%M:%S')

    # Build the flow filter from the 5-tuple when ports are available.
    # Get the protocol number.
    proto_num = int(flow['Protocol'])
    src_ip, dst_ip = flow['Source IP'], flow['Destination IP']
    src_port, dst_port = int(flow['Source Port']), int(flow['Destination Port'])
    
    # Build the BPF filter.
    # For TCP (6) or UDP (17), use the 5-tuple filter with ports.
    if proto_num in [6, 17]:
        bpf = (f"host {src_ip} and host {dst_ip} and "
               f"port {src_port} and port {dst_port} and proto {proto_num}")
    else:
        # For ICMP (1) or protocols without ports, filter by IP pair and protocol only.
        bpf = f"host {src_ip} and host {dst_ip} and proto {proto_num}"

    tmp_pcap, res_pcap = f"tmp_{idx}.pcap", f"res_{idx}.pcap"
    
    # Run packet extraction commands.
    subprocess.run(f'TZ=UTC editcap -A "{start_w}" -B "{end_w}" {PCAP_FILE} {tmp_pcap}', shell=True, capture_output=True)
    subprocess.run(f'tcpdump -r {tmp_pcap} "{bpf}" -w {res_pcap}', shell=True, capture_output=True)

    result = None
    if os.path.exists(res_pcap):
        extracted_data = process_single_flow(res_pcap, flow['Source IP'])
        if extracted_data:
            result = {
                'label': flow['Label'],
                'timestamp': raw_timestamp,
                'flow_id': f"{flow['Source IP']}-{flow['Destination IP']}-{flow['Source Port']}-{flow['Destination Port']}",
                'packets': extracted_data
            }
        # Clean up temporary files.
        if os.path.exists(tmp_pcap): os.remove(tmp_pcap)
        if os.path.exists(res_pcap): os.remove(res_pcap)
    
    return result

def main():
    df = pd.read_csv(CSV_FILE, encoding = 'cp1252')  # Handle possible encoding issues.
    df.columns = df.columns.str.strip()
    df_sampled = df.groupby('Label').head(samples_per_label).reset_index(drop=True)     # Limit sample count.
    # df_sampled = df  # Process the full dataset.
    
    print(f"Starting multiprocessing. Workers: {NUM_WORKERS} | Total flows: {len(df_sampled)}")

    current_batch = []
    chunk_id = 0
    
    # Run tasks in parallel with a process pool.
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks.
        futures = {executor.submit(worker_task, i, row): i for i, row in df_sampled.iterrows()}
        
        # Track completion progress with tqdm.
        for i, future in enumerate(tqdm(as_completed(futures), total=len(df_sampled), desc="Processing")):
            res = future.result()
            if res:
                current_batch.append(res)

            # Save chunks.
            if (len(current_batch)) >= SAVE_INTERVAL:
                chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}.pkl")
                with open(chunk_file, 'wb') as f:
                    pickle.dump(current_batch, f)
                tqdm.write(f"Saved chunk {chunk_id} (completed {i+1} rows)")
                current_batch = []
                chunk_id += 1

    # Save the final partial batch.
    if current_batch:
        chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}.pkl")
        with open(chunk_file, 'wb') as f:
            pickle.dump(current_batch, f)
        print(f"Processing complete. Saved {chunk_id + 1} chunks.")

if __name__ == "__main__":
    main()