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

import requests
from django.core.files.storage import default_storage

def download_temp_file(django_file):
    """
    Return a BytesIO containing the file contents.
    Works for cloudinary (http(s) URLs) and for storage backends (default_storage).
    """
    from io import BytesIO

    # If django_file.url is an absolute URL (cloudinary), fetch it
    url = getattr(django_file, "url", None)
    if url and (url.startswith("http://") or url.startswith("https://")):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            buf = BytesIO(resp.content)
            buf.seek(0)
            return buf
        except Exception as e:
            raise RuntimeError(f"Could not fetch HTTP file: {e}")

    # Otherwise, try reading using Django storage backend (works for local and many storages)
    try:
        name = getattr(django_file, "name", None)
        if not name:
            raise RuntimeError("File has no name and is not an HTTP URL.")
        with default_storage.open(name, "rb") as fh:
            content = fh.read()
        buf = BytesIO(content)
        buf.seek(0)
        return buf
    except Exception as e:
        raise RuntimeError(f"Could not fetch file from storage: {e}")



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

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

@login_required
def upload_view(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]

            # save via default_storage to force the active storage backend (Cloudinary)
            try:
                # If 'f' is an UploadedFile we can pass its chunks to default_storage
                saved_name = default_storage.save(f.name, ContentFile(f.read()))
            except Exception as e:
                # fallback: try using default create (old behaviour) and report error
                import traceback
                traceback.print_exc()
                messages.error(request, f"Upload failed while saving file: {e}")
                return redirect("core:upload")

            try:
                obj = UploadedFile.objects.create(
                    owner=request.user,
                    file=saved_name,      # store the saved path returned by storage.save()
                    filename=f.name
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                messages.error(request, f"Upload failed (DB save): {e}")
                return redirect("core:upload")

            # debug: log the resulting URL (so you can verify instantly in logs)
            try:
                print("DEBUG: uploaded file saved_name=", saved_name, "url=", obj.file.url)
            except Exception:
                pass

            messages.success(request, "File uploaded. Generating EDA...")
            # optional: run EDA immediately and save summary
            try:
                eda = generate_eda(obj, request)
                obj.eda_summary = eda or {}
                obj.save()
            except Exception as e:
                messages.warning(request, f"EDA failed: {e}")
                obj.eda_summary = {}
                obj.save()
            return redirect("core:file_detail", pk=obj.pk)
    else:
        form = UploadForm()
    return render(request, "core/upload.html", {"form": form})



@login_required
def file_detail_view(request, pk):
    obj = get_object_or_404(UploadedFile, pk=pk, owner=request.user)
    eda = obj.eda_summary or {}
    charts = eda.get("charts", [])
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
            buffer = download_temp_file(obj.file)
            df = pd.read_csv(buffer)
        elif ext in ("xls", "xlsx"):
            buffer = download_temp_file(obj.file)
            df = pd.read_excel(buffer)
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
            buffer = download_temp_file(obj.file)
            df = pd.read_csv(buffer, nrows=200)
            file_text = df.to_csv(index=False)
        elif ext in ("xls", "xlsx"):
            buffer = download_temp_file(obj.file)
            df = pd.read_excel(buffer, nrows=200)
            file_text = df.to_csv(index=False)
        elif ext == "pdf":
            from pdfminer.high_level import extract_text
            buffer = download_temp_file(obj.file)
            extracted = extract_text(buffer)
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


import json
import numpy as np
import pandas as pd
from io import BytesIO
from django.conf import settings
from pdfminer.high_level import extract_text

# reuse convert_json (or include it if not present)
def convert_json(o):
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        # pandas NA
        import pandas as _pd
        if _pd.isna(o):
            return None
    except Exception:
        pass
    return str(o)

def generate_eda(uploaded_obj, request=None):
    """
    Lightweight, safe EDA for production.
    Returns a JSON-serializable dict with:
      - shape, columns, head (first 5 rows),
      - basic_eda: dtypes, missing counts, describe (limited),
      - charts: list of Cloudinary URLs for histograms (up to 3 numeric cols),
      - errors: optional error message
    The function will never raise; on unrecoverable errors it returns {'error': '...'}.
    """
    try:
        # Use helper if available to retrieve file into BytesIO (works for local/cloud URLs)
        try:
            buffer = None
            # prefer using download_temp_file if defined in the module
            if 'download_temp_file' in globals() and callable(download_temp_file):
                buffer = download_temp_file(uploaded_obj.file)
            else:
                # fallback: try reading via url with requests (short timeout)
                import requests
                resp = requests.get(uploaded_obj.file.url, timeout=10)
                buffer = BytesIO(resp.content)
            buffer.seek(0)
        except Exception as e:
            return {"error": f"Could not fetch file: {e}"}

        ext = uploaded_obj.filename.lower().split(".")[-1]

        # Limit rows/load to avoid memory spike
        nrows_preview = 2000
        simple_df = None

        if ext in ("csv", "txt"):
            try:
                # try fast read first, then fallback to python engine on failure
                try:
                    df = pd.read_csv(buffer, nrows=nrows_preview, low_memory=True)
                except Exception:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, nrows=nrows_preview, engine="python", on_bad_lines="skip")
                simple_df = df
            except Exception as e:
                return {"error": f"CSV read error: {e}"}

        elif ext in ("xls", "xlsx"):
            try:
                buffer.seek(0)
                df = pd.read_excel(buffer, nrows=nrows_preview, engine="openpyxl")
                simple_df = df
            except Exception as e:
                return {"error": f"Excel read error: {e}"}

        elif ext == "pdf":
            try:
                buffer.seek(0)
                text = extract_text(buffer) or ""
                # small dataframe with a preview of text
                simple_df = pd.DataFrame({"document_content": [text[:10000]]})
            except Exception as e:
                return {"error": f"PDF read error: {e}"}
        else:
            return {"error": "Unsupported file format for EDA"}

        # ---------- BASIC SUMMARY ----------
        try:
            # head (limit to 5 rows and small strings)
            head_rows = simple_df.head(5).copy()
            for col in head_rows.select_dtypes(include=["object"]).columns:
                head_rows[col] = head_rows[col].astype(str).str.slice(0, 200)

            basic_eda = {
                "dtypes": simple_df.dtypes.astype(str).to_dict(),
                "missing": simple_df.isnull().sum().to_dict(),
                "describe_numeric": simple_df.select_dtypes(include="number").describe().to_dict()
                                    if not simple_df.select_dtypes(include="number").empty else {},
                "describe_object": simple_df.select_dtypes(include="object").describe().to_dict()
                                    if not simple_df.select_dtypes(include="object").empty else {},
            }

            # --- Merge describe results (outside the dictionary) ---
            try:
                merged = {}

                # numeric first
                for col, stats in basic_eda["describe_numeric"].items():
                    merged[col] = stats

                # object next
                for col, stats in basic_eda["describe_object"].items():
                    merged[col] = stats

                # collect headers dynamically
                headers = set()
                for stats in merged.values():
                    headers.update(stats.keys())

                basic_eda["describe"] = merged
                basic_eda["describe_headers"] = list(headers)

            except Exception:
                basic_eda["describe"] = {}
                basic_eda["describe_headers"] = []

        except Exception as e:
            basic_eda = {"error": f"Basic EDA failed: {e}"}

        # ---------- QUICK CHARTS (up to 3 numeric columns) ----------
        charts_urls = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            numeric_cols = list(simple_df.select_dtypes(include="number").columns)[:3]
            for i, col in enumerate(numeric_cols):
                try:
                    vals = simple_df[col].dropna()
                    if vals.shape[0] == 0:
                        continue
                    plt.figure(figsize=(4,3))
                    # small histogram; limited bins
                    vals.hist(bins=20)
                    plt.title(f"{col} distribution")
                    plt.tight_layout()
                    fig = plt.gcf()

                    # save to BytesIO
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=100)
                    plt.close(fig)
                    buf.seek(0)

                    # upload via helper (uses cloudinary.uploader.upload inside)
                    try:
                        public_id = f"file_{uploaded_obj.id}_hist_{i}"
                        upload = cloudinary.uploader.upload(buf, folder="charts", public_id=public_id, resource_type="image")
                        charts_urls.append(upload.get("secure_url"))
                    except Exception:
                        # fallback: do not crash charts step
                        pass
                except Exception:
                    # continue generating other charts
                    continue
        except Exception:
            charts_urls = []

                # ---------- OPTIONAL YDATA REPORT (lightweight) ----------
        report_url = None
        try:
            from ydata_profiling import ProfileReport
            profile = ProfileReport(
                simple_df,
                minimal=True,          # lightweight mode
                explorative=False,
                correlations={"auto": {"calculate": False}},
            )

            html_buffer = BytesIO()
            tmp_path = f"/tmp/report_{uploaded_obj.id}.html"
            profile.to_file(tmp_path)

            with open(tmp_path, "rb") as f:
                upload = cloudinary.uploader.upload(
                    f,
                    folder="reports",
                    resource_type="raw",
                    public_id=f"report_{uploaded_obj.id}.html",   # IMPORTANT FIX
                    format="html"
                )

            report_url = upload.get("secure_url")
        except Exception as e:
            report_url = None


        # ---------- FINAL SUMMARY ----------
        summary = {
            "shape": tuple(simple_df.shape),
            "columns": list(map(str, simple_df.columns)),
            "head": json.loads(head_rows.to_json(orient="split")) if 'head_rows' in locals() else {},
            "basic_eda": basic_eda,
            "charts": [u for u in charts_urls if u],
            "report_url": report_url,
        }

        # ensure JSON-safe
        try:
            summary = json.loads(json.dumps(summary, default=convert_json))
        except Exception:
            # final fallback: minimal safe summary
            return {
                "shape": summary.get("shape"),
                "columns": summary.get("columns"),
                "basic_eda": {},
                "charts": summary.get("charts", []),
            }

        return summary

    except Exception as e:
        # absolutely never bubble up
        return {"error": f"Unhandled EDA error: {e}"}

@user_passes_test(lambda u: u.is_staff)
def admin_dashboard_view(request):
    total_users = __import__("django.contrib.auth").contrib.auth.get_user_model().objects.count()
    total_uploads = UploadedFile.objects.count()
    recent = UploadedFile.objects.order_by("-uploaded_at")[:10]
    return render(request, "core/admin_dashboard.html", {"total_users": total_users, "total_uploads": total_uploads, "recent": recent})
