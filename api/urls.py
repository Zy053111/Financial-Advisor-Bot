from django.urls import path
from .views import RebalanceRecommendationView, BacktestHistoryView, PortfolioChatView

urlpatterns = [
    path('recommendation/', RebalanceRecommendationView.as_view(), name='rebalance-recommendation'),
    path('backtest-history/', BacktestHistoryView.as_view(), name='backtest-history'),
    path('chat/', PortfolioChatView.as_view(), name='portfolio-chat'),
]