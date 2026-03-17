from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os, json, base64, cv2, torch
from torchvision import transforms, models
from PIL import Image
import traceback
import numpy as np

# ========== CONFIG ==========
YOLO_MODEL_PATH = "best.pt"
VIT_MODEL_PATH = "vit_tiny_fruit_disease.pth"
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
YOLO_CONF_THRESHOLD = 0.25   # Detection sensitivity
FUSION_WEIGHT = 0.5          # Model fusion balance (0 = only ViT, 1 = only YOLO)

# ========== APP SETUP ==========
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ========== LOAD MODELS ==========
print("🔹 Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("🔹 Loading ViT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = models.vit_b_16(weights=None)
num_classes = 59  # your number of disease classes
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)
vit_model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
vit_model.to(device)
vit_model.eval()

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load disease info
with open("diseases.json", "r", encoding="utf-8") as f:
    DISEASE_INFO = json.load(f)
ALL_LABELS = list(DISEASE_INFO.keys())

# ========== CPM SIMPLE PRIOR TABLE ==========
CPM_PRIORS = {
    "Tomato": {
        "Anthracnose_tomato": 1.2,
        "Blossom end rot": 1.1
    },
    "Apple": {
        "rust": 1.2,
        "HEALTHY_APPLE": 1.0
    },
    "Peach": {
        "bacterial_spot_peach": 1.3
    }
}

@app.route("/")
def home():
    return jsonify({"status": "Hybrid Model Active", "labels": ALL_LABELS[:5]})

# ========== PREDICT ==========

import math

def hybrid_decision(yolo_label, yolo_conf, vit_logits, slider_weight):

    # Softmax probabilities
    probs = torch.softmax(vit_logits, dim=1)
    probs_np = probs.cpu().numpy()[0]

    # Get top prediction
    vit_conf = float(torch.max(probs).item())
    vit_idx = int(torch.argmax(probs).item())
    vit_label = ALL_LABELS[vit_idx] if vit_idx < len(ALL_LABELS) else yolo_label

    # --------- ENTROPY CALCULATION ----------
    epsilon = 1e-10  # avoid log(0)
    entropy = -sum(p * math.log(p + epsilon) for p in probs_np)

    max_entropy = math.log(len(probs_np))
    normalized_entropy = entropy / max_entropy

    stability_score = 1 - normalized_entropy

    # Recalibrated confidence
    adjusted_vit_conf = vit_conf * stability_score

    print("\n🔍 HYBRID CHECK WITH ENTROPY")
    print(f"YOLO → {yolo_label} ({yolo_conf:.2f})")
    print(f"ViT Raw → {vit_label} ({vit_conf:.2f})")
    print(f"Entropy → {entropy:.4f}")
    print(f"Stability → {stability_score:.4f}")
    print(f"Adjusted ViT → {adjusted_vit_conf:.4f}")

    # --------- FUSION LOGIC ----------
    if vit_label == yolo_label:
        final_label = vit_label
        final_conf = (yolo_conf + adjusted_vit_conf) / 2
        decision = "MATCH"
    else:
        yolo_score = slider_weight * yolo_conf
        vit_score = (1 - slider_weight) * adjusted_vit_conf

        if yolo_score >= vit_score:
            final_label = yolo_label
            final_conf = yolo_conf
        else:
            final_label = vit_label
            final_conf = adjusted_vit_conf

        decision = "FUSION"

    print(f"✅ FINAL → {final_label} ({final_conf:.2f}) | Decision: {decision}\n")

    return final_label, final_conf, decision, entropy, stability_score


def apply_cpm(label, confidence):

    crop = DISEASE_INFO.get(label, {}).get("crop", None)

    if crop in CPM_PRIORS:
        weight = CPM_PRIORS[crop].get(label, 0.8)
    else:
        weight = 0.9

    new_conf = confidence * weight
    new_conf = min(new_conf, 1.0)

    print(f"🌱 CPM → Crop: {crop}, Weight: {weight}, New Conf: {new_conf:.4f}")

    return new_conf



def append_severity_action(text_dict, severity_stage):

    severity_messages = {
        "Mild": {
            "en": "• Early stage detected. Prefer organic control and monitor crop regularly.",
            "hi": "• प्रारंभिक अवस्था पाई गई। जैविक नियंत्रण का उपयोग करें और नियमित निगरानी करें।",
            "te": "• ప్రారంభ దశ గుర్తించబడింది. సేంద్రీయ నియంత్రణను ఉపయోగించి పంటను పర్యవేక్షించండి."
        },
        "Moderate": {
            "en": "• Moderate infection. Combine chemical and cultural control methods.",
            "hi": "• मध्यम संक्रमण। रासायनिक और सांस्कृतिक नियंत्रण विधियों का संयोजन करें।",
            "te": "• మధ్యస్థ స్థాయి సంక్రమణ. రసాయన మరియు సాంస్కృతిక నియంత్రణ పద్ధతులను ఉపయోగించండి."
        },
        "Severe": {
            "en": "• Severe infection detected. Immediate intensive treatment required. Consider isolating infected plants.",
            "hi": "• गंभीर संक्रमण पाया गया। तुरंत गहन उपचार आवश्यक है। संक्रमित पौधों को अलग करें।",
            "te": "• తీవ్రమైన సంక్రమణ గుర్తించబడింది. తక్షణమే తీవ్రమైన చికిత్స అవసరం. సోకిన మొక్కలను వేరుచేయండి."
        }
    }

    if not isinstance(text_dict, dict):
        return text_dict

    updated = {}

    for lang in text_dict:
        base_text = text_dict.get(lang, "")
        extra = severity_messages.get(severity_stage, {}).get(lang, "")
        updated[lang] = base_text + "\n\n" + extra

    return updated


def scale_treatment_by_severity(base_treatment, severity_stage):
    return append_severity_action(base_treatment, severity_stage)


# ========== PREDICT ==========
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files['image']
        filename = file.filename or "upload.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)

        # ✅ RESTORE CONFIDENCE SLIDER LOGIC

        # ---------- Detection Threshold ----------
        yolo_conf_input = request.form.get("yolo_threshold", None)
        if yolo_conf_input is not None:
            yolo_threshold = float(yolo_conf_input)
        else:
            yolo_threshold = YOLO_CONF_THRESHOLD

        # ---------- Fusion Weight ----------
        fusion_input = request.form.get("fusion_weight", None)
        if fusion_input is not None:
            fusion_weight = float(fusion_input)
        else:
            fusion_weight = FUSION_WEIGHT

        print(f"🔸 YOLO Threshold: {yolo_threshold}")
        print(f"🔸 Fusion Weight: {fusion_weight}")

        results = yolo_model(img_path, conf=yolo_threshold)
        img = cv2.imread(img_path)

        image_height, image_width = img.shape[:2]
        image_area = image_height * image_width

        infection_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        box_count = 0

        detections = []
        disease_counts = {}
        crop_counts = {}
        detected_diseases = set()

        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                yolo_label = names.get(cls_id, "Unknown")
                yolo_conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                infection_mask[y1:y2, x1:x2] = 1
                box_count += 1
                crop_img = img[y1:y2, x1:x2]

                if crop_img.size == 0:
                    continue

                pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                input_tensor = vit_transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    vit_logits = vit_model(input_tensor)

                final_label, final_conf, decision_type, entropy, stability = hybrid_decision(
                    yolo_label, yolo_conf, vit_logits, fusion_weight
                )

                # APPLY CPM
                final_conf = apply_cpm(final_label, final_conf)

                detections.append({
                    "label": final_label,
                    "confidence": float(final_conf),
                    "decision_type": decision_type,
                    "entropy": float(entropy),
                    "stability_score": float(stability),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

                detected_diseases.add(final_label)
                disease_counts[final_label] = disease_counts.get(final_label, 0) + 1
                crop = DISEASE_INFO.get(final_label, {}).get("crop", "Unknown")
                crop_counts[crop] = crop_counts.get(crop, 0) + 1

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                text = f"{final_label} ({final_conf*100:.1f}%)"
                cv2.putText(img, text, (x1, max(y1-5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        


        _, buffer = cv2.imencode(".jpg", img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")


        # --------- IDGI SEVERITY CALCULATION ----------
        if box_count > 0:

            infected_pixels = np.sum(infection_mask)
            area_ratio = infected_pixels / image_area  # % infected area

            lesion_density = box_count / image_area   # lesion count per pixel

            # ---- Compute lesion center points ----
            lesion_centers = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    lesion_centers.append((cx, cy))

            # ---- Compute dispersion score ----
            dispersion_score = 0.0

            if len(lesion_centers) > 1:
                distances = []

                for i in range(len(lesion_centers)):
                    for j in range(i + 1, len(lesion_centers)):
                        dx = lesion_centers[i][0] - lesion_centers[j][0]
                        dy = lesion_centers[i][1] - lesion_centers[j][1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        distances.append(dist)

                avg_distance = sum(distances) / len(distances)

                # normalize by image diagonal
                image_diagonal = math.sqrt(image_width**2 + image_height**2)
                dispersion_score = avg_distance / image_diagonal

            # ---- Growth Spread Factor ----
            spread_factor = area_ratio * dispersion_score

            # ---- FINAL IDGI ----
            # Normalize lesion density to actual box count
            normalized_density = box_count / 10.0  # scaling factor for interpretability

            idgi = (
                (area_ratio * 0.5) +
                (normalized_density * 0.2) +
                (dispersion_score * 0.3)
            )

            # --------- SDTS (Self Dynamic Thresholding System) ---------

            # calculate average stability from detections
            avg_stability = (
                sum(float(d["stability_score"]) for d in detections) / len(detections)
            ) if detections else 0.5

            # dynamic threshold adjustment
            mild_threshold = 0.20 + (1 - avg_stability) * 0.1
            moderate_threshold = 0.45 + (1 - avg_stability) * 0.1

            # adjust if many lesions
            if box_count > 2:
                mild_threshold -= 0.05
                moderate_threshold -= 0.05

            # apply severity stage
            if idgi < mild_threshold:
                severity_stage = "Mild"
            elif idgi < moderate_threshold:
                severity_stage = "Moderate"
            else:
                severity_stage = "Severe"

            print(f"🧠 SDTS → Stability: {avg_stability:.4f}")
            print(f"🧠 Dynamic Mild Threshold: {mild_threshold:.4f}")
            print(f"🧠 Dynamic Moderate Threshold: {moderate_threshold:.4f}")

            
            

        else:
            area_ratio = 0.0
            lesion_density = 0.0
            dispersion_score = 0.0
            idgi = 0.0
            severity_stage = "None"



            # ✅ RESTORE DISEASE INFO
        diseases_info = {}
        for d in detected_diseases:
            info = DISEASE_INFO.get(d, {})
            yt_query = info.get("youtube_query") or f"{info.get('crop','')} {d} disease"
            diseases_info[d] = {
                "crop": info.get("crop", "Unknown"),
                "display_name": info.get("display_name", {}),
                "cause": info.get("cause", {}),
                "treatment": scale_treatment_by_severity(
                    info.get("treatment", {}),
                    severity_stage
                ),
                "prevention": info.get("prevention", {}),
                "youtube_link": "https://www.youtube.com/results?search_query=" + "+".join(yt_query.split())
            }

        

            

        print("\n📊 IDGI SEVERITY ANALYSIS")
        print(f"Area Ratio → {area_ratio:.4f}")
        print(f"Lesion Density → {lesion_density:.6f}")
        print(f"Dispersion Score → {dispersion_score:.4f}")
        print(f"IDGI Score → {idgi:.4f}")
        print(f"Severity Stage → {severity_stage}\n")

        # --------- OVERALL STABILITY ----------
        avg_stability = float(
            sum(float(d["stability_score"]) for d in detections) / len(detections)
            ) if detections else 0.0

        return jsonify({
            "message": "Detections found" if detections else "No detections",
            "image": encoded_image,
            "detections": detections,
            "total_detections": len(detections),
            "average_stability": avg_stability,

            "confidence_threshold": yolo_threshold,
            "fusion_weight": fusion_weight,

            "idgi_score": float(idgi),
            "dispersion_score": float(dispersion_score),
            "severity_stage": severity_stage,
            "area_ratio": float(area_ratio),
            "lesion_density": float(lesion_density),

            "disease_counts": disease_counts,
            "crop_counts": crop_counts,
            "disease_info": diseases_info
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask Hybrid Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)