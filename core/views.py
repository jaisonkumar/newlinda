import os
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from .forms import UploadForm, RegisterForm
from .models import UploadedFile
from django.contrib import messages
from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm

from io import BytesIO
import cloudinary.uploader

def upload_chart_to_cloudinary(fig, public_id):
    """
    Saves a Matplotlib figure to Cloudinary and returns the secure URL.
    """
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    buffer.seek(0)

    upload = cloudinary.uploader.upload(
        buffer,
        folder="charts",
        public_id=public_id,
        resource_type="image"
    )

    return upload["secure_url"]


import requests
from django.core.files.storage import default_storage

def download_temp_file(django_file):
    """
    Downloads a Cloudinary file to a temporary local file path.
    Returns the local file path.
    """
    url = django_file.url
    resp = requests.get(url)

    temp_path = os.path.join(settings.MEDIA_ROOT, "temp", os.path.basename(url))
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(resp.content)

    return temp_path


# data libs
import pandas as pd
import matplotlib.pyplot as plt

# For PDF text extraction
try:
    import textract
except Exception:
    textract = None

# For Gemini / Google Gen AI
# Note: adapt the exact import to the SDK you choose. Example below is illustrative.
# Install and follow official Gen AI / Gemini docs (library and method names may change).
try:
    from google import genai
except Exception:
    genai = None

def home_view(request):
    files = None
    if request.user.is_authenticated:
        files = request.user.uploads.all().order_by("-uploaded_at")
    return render(request, "core/home.html", {"files": files})


from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.shortcuts import redirect, render

def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "🎉 Account created successfully! You can now log in.")
            return redirect("core:login")
        else:
            messages.error(request, "There was an error. Please correct the highlighted fields.")
    else:
        form = UserCreationForm()

    return render(request, "core/register.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("core:home")
        messages.error(request, "Invalid credentials.")
    return render(request, "core/login.html")

def logout_view(request):
    logout(request)
    return redirect("core:home")

@login_required
def upload_view(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]
            obj = UploadedFile.objects.create(owner=request.user, file=f, filename=f.name)
            messages.success(request, "File uploaded. Generating EDA...")
            # optional: run EDA immediately and save summary
            try:
                eda = generate_eda(obj, request)
                obj.eda_summary = eda
                obj.save()
            except Exception as e:
                messages.warning(request, f"EDA generation failed: {e}")
            return redirect("core:file_detail", pk=obj.pk)
    else:
        form = UploadForm()
    return render(request, "core/upload.html", {"form": form})

@login_required
def file_detail_view(request, pk):
    obj = get_object_or_404(UploadedFile, pk=pk, owner=request.user)
    eda = obj.eda_summary or {}
    charts = obj.eda_summary.get("charts", [])
    return render(request, "core/file_detail.html", {"file": obj, "eda": eda, "charts": charts})

import pandas as pd
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
import json

def query_file_view(request, pk):
    obj = get_object_or_404(UploadedFile, pk=pk, owner=request.user)

    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    user_query = json.loads(request.body.decode("utf-8")).get("query", "")
    if not user_query:
        return JsonResponse({"error": "Empty query"}, status=400)

    # Load dataset
    ext = obj.filename.lower().split(".")[-1]
    try:
        if ext in ("csv", "txt"):
            df = pd.read_csv(obj.file.path)
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(obj.file.path)
        else:
            return JsonResponse({"error": "Query not supported for this file type"})
    except Exception as e:
        return JsonResponse({"error": str(e)})

    # Ask AI to convert text to valid Pandas query
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)

        prompt = f"""
Convert the following natural language condition into a valid Pandas DataFrame.query() expression.
DO NOT return explanation. ONLY return the query code.

User text:
{user_query}

Columns available:
{list(df.columns)}

Examples:
"age < 30" → age < 30
"gender is male" → Gender == 'Male'
"high income urban customers" → IncomeLevel == "High" and Location == "Urban"
"""

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        pandas_query = response.text.strip()

    except Exception as e:
        return JsonResponse({"error": f"AI failed: {e}"})


    # Run the query safely
    try:
        result_df = df.query(pandas_query)
    except Exception as e:
        return JsonResponse({"error": f"Query Error: {str(e)}"})

    # Convert to HTML table
    html = result_df.head(50).to_html(classes="eda-table", border=0)

    return JsonResponse({
        "query": pandas_query,
        "rows": len(result_df),
        "html": f"<strong>Query:</strong> {pandas_query}<br><strong>Rows:</strong> {len(result_df)}<br>{html}"
    })


@login_required
def ask_file_view(request, pk):
    """
    Enhanced AI view: Sends file preview + Basic EDA (dtypes, missing, describe) to Gemini
    """
    obj = get_object_or_404(UploadedFile, pk=pk, owner=request.user)

    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    data = json.loads(request.body.decode("utf-8"))
    question = data.get("question", "").strip()

    if not question:
        return JsonResponse({"error": "Empty question"}, status=400)

    # -----------------------------
    # LOAD BASIC EDA FROM DB
    # -----------------------------
    eda = obj.eda_summary or {}
    basic = eda.get("basic_eda", {})

    dtypes = basic.get("dtypes", {})
    missing = basic.get("missing", {})
    describe = basic.get("describe", {})

    # -----------------------------
    # SAFE FILE PREVIEW
    # -----------------------------
    ext = obj.filename.lower().split(".")[-1]
    file_text = ""
    
    try:
        if ext in ("csv", "txt"):
            df = pd.read_csv(obj.file.path, nrows=200)
            file_text = df.to_csv(index=False)
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(obj.file.path, nrows=200)
            file_text = df.to_csv(index=False)
        elif ext == "pdf":
            from pdfminer.high_level import extract_text
            extracted = extract_text(obj.file.path)
            file_text = extracted[:20000] if extracted else "No readable text extracted."
        else:
            file_text = "Unsupported file type for preview."
    except Exception as e:
        file_text = f"Error reading file: {e}"

    # -----------------------------
    # BUILD THE SUPER-PROMPT
    # -----------------------------
    prompt = f"""
You are a highly intelligent data analysis assistant.

The user uploaded a file. Below is:
1. A preview of the file contents
2. Column Data Types
3. Missing Values
4. Describe Statistics (mean, std, min, max, percentiles)

Use all this information to answer the user's question clearly and accurately.
Remember to generate plain text answers without any special formatting like ** or anything similar
--------------------
📄 FILE PREVIEW (first rows):
{file_text}

--------------------
📘 COLUMN DATA TYPES:
{json.dumps(dtypes, indent=2)}

--------------------
🟨 MISSING VALUES:
{json.dumps(missing, indent=2)}

--------------------
📊 DESCRIBE STATISTICS:
{json.dumps(describe, indent=2)}

--------------------
❓ USER QUESTION:
{question}

When referencing columns, always use exact column names.
If data suggests patterns, highlight them.
Provide short but accurate answers.
"""

    # -----------------------------
    # CALL GEMINI
    # -----------------------------
    answer = "AI not configured."

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        answer = response.text

    except Exception as e:
        answer = f"AI error: {e}"

    return JsonResponse({"answer": answer})

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from django.conf import settings
from pdfminer.high_level import extract_text
from ydata_profiling import ProfileReport


def convert_json(o):
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if pd.isna(o):
        return None
    return str(o)


def generate_eda(uploaded_obj, request=None):
    """
    Enhanced EDA using:
    - ydata_profiling for full HTML report
    - pdfminer.six for PDF text extraction
    - Matplotlib numeric distribution charts
    - Basic EDA summary (dtypes, missing values, describe stats)
    """

    file_path = uploaded_obj.file.path
    ext = uploaded_obj.filename.lower().split(".")[-1]

    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.MEDIA_ROOT, "charts"), exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣ HANDLE PDF → EXTRACT TEXT → CREATE DATAFRAME
    # ------------------------------------------------------------------
    if ext == "pdf":
        try:
            try:
                text = extract_text(file_path)
            except Exception:
                text = ""
            df = pd.DataFrame({"document_content": [text]})
        except Exception as e:
            return {"error": f"PDF extraction error: {str(e)}"}

    # ------------------------------------------------------------------
    # 2️⃣ HANDLE CSV / TXT
    # ------------------------------------------------------------------
    elif ext in ("csv", "txt"):
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
        except Exception as e:
            return {"error": f"CSV/TXT read error: {str(e)}"}

    # ------------------------------------------------------------------
    # 3️⃣ HANDLE EXCEL
    # ------------------------------------------------------------------
    elif ext in ("xls", "xlsx"):
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception as e:
            return {"error": f"Excel read error: {str(e)}"}

    else:
        return {"error": "Unsupported file format for EDA"}

    # ------------------------------------------------------------------
    # 4️⃣ GENERATE YDATA PROFILING REPORT (HTML)
    # ------------------------------------------------------------------
    try:
        profile = ProfileReport(df, title="Report", explorative=True)
        report_path = os.path.join(settings.REPORTS_DIR, f"report_{uploaded_obj.id}.html")
        profile.to_file(report_path)

    except Exception as e:
        print("Profiling error:", e)
        report_path = None

    # ------------------------------------------------------------------
    # 5️⃣ SAVE SIMPLE CHARTS (FIRST 4 NUMERIC COLUMNS)
    # ------------------------------------------------------------------
    charts_urls = []
    numeric_cols = df.select_dtypes(include="number").columns[:4]

    for i, col in enumerate(numeric_cols):
        try:
            plt.figure(figsize=(6, 4))
            df[col].dropna().hist()
            plt.title(f"{col} distribution")
            public_id = f"file_{uploaded_obj.id}_hist_{i}"
            fig = plt.gcf()  # get current figure
            chart_url = upload_chart_to_cloudinary(fig, public_id)
            plt.close()
            charts_urls.append(chart_url)
        except:
            print("Chart error:", e)
            pass

    # ------------------------------------------------------------------
    # 6️⃣ BASIC EDA SUMMARY (NEW)
    # ------------------------------------------------------------------
    try:
        basic_eda = {
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "describe": df.describe(include="all").fillna("").to_dict()
        }
        try:
            first_col = next(iter(basic_eda["describe"].values()))
            basic_eda["describe_headers"] = list(first_col.keys())
        except:
            basic_eda["describe_headers"] = []
    except Exception as e:
        basic_eda = {"error": f"Basic EDA generation error: {e}"}

    # ------------------------------------------------------------------
    # 7️⃣ FINAL JSON-SAFE SUMMARY FOR UI
    # ------------------------------------------------------------------
    summary = {
        "shape": df.shape,
        "columns": list(df.columns.astype(str)),
        "head": df.head(5).to_dict(),
        "basic_eda": basic_eda,
        "charts": charts_urls,
        "report_url": settings.MEDIA_URL + f"reports/report_{uploaded_obj.id}.html"
                        if report_path else None
    }

    summary = json.loads(json.dumps(summary, default=convert_json))
    return summary

@user_passes_test(lambda u: u.is_staff)
def admin_dashboard_view(request):
    total_users = __import__("django.contrib.auth").contrib.auth.get_user_model().objects.count()
    total_uploads = UploadedFile.objects.count()
    recent = UploadedFile.objects.order_by("-uploaded_at")[:10]
    return render(request, "core/admin_dashboard.html", {"total_users": total_users, "total_uploads": total_uploads, "recent": recent})
