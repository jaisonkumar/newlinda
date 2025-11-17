from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = "core"

urlpatterns = [
    path("", views.home_view, name="home"),
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_view, name="upload"),
    path("files/<int:pk>/", views.file_detail_view, name="file_detail"),
    path("files/<int:pk>/ask/", views.ask_file_view, name="ask_file"),
    path("admin-dashboard/", views.admin_dashboard_view, name="admin_dashboard"),
    path("files/<int:pk>/query/", views.query_file_view, name="query_file"),
]

# 🔥 Add this for file uploads (media files)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
