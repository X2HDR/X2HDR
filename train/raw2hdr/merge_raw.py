import HDRutils
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from src.hdr_utils import save_hdr_image
from tqdm import tqdm
import time

def process_single_folder(root_folder):
    start_time = time.time()
    
    if os.path.isfile(root_folder):
        return {"folder": root_folder, "status": "skipped", "reason": "is_file", "time": 0}
    
    try:
        files = [f for f in os.listdir(root_folder) if f.endswith('.NEF')]
        if not files:
            return {"folder": root_folder, "status": "skipped", "reason": "no_NEF_files", "time": time.time() - start_time}
        
        files = [os.path.join(root_folder, file) for file in files]
        HDR_img = HDRutils.merge(files, do_align=True, estimate_exp='mst')[0]
        output_path = os.path.join(root_folder, 'merged_hdr.exr')
        save_hdr_image(HDR_img, output_path)
        
        return {"folder": root_folder, "status": "success", "files_count": len(files), "time": time.time() - start_time}
    
    except Exception as e:
        return {"folder": root_folder, "status": "error", "error": str(e), "time": time.time() - start_time}


root_folders = [os.path.join('HDRPS_Raws', folder) for folder in os.listdir('HDRPS_Raws')]

max_workers = 4
successful = 0
failed = 0
skipped = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    with tqdm(total=len(root_folders), desc="Processing folders") as pbar:
        future_to_folder = {executor.submit(process_single_folder, folder): folder 
                            for folder in root_folders}
        
        for future in concurrent.futures.as_completed(future_to_folder):
            result = future.result()
            pbar.update(1)
            
            if result["status"] == "success":
                successful += 1
                print(f"✅ {os.path.basename(result['folder'])}: {result['files_count']} files, {result['time']:.2f}s")
            elif result["status"] == "error":
                failed += 1
                print(f"❌ {os.path.basename(result['folder'])}: {result['error']}")
            else:
                skipped += 1
                print(f"⏭️  {os.path.basename(result['folder'])}: {result['reason']}")

print(f"\n📊 Summary: {successful} successful, {failed} failed, {skipped} skipped")