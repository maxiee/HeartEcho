import 'dart:async';

import 'package:app/api/api.dart';
import 'package:app/models/training_session.dart';
import 'package:flutter/foundation.dart';

class GlobalTrainingSessionProvider extends ChangeNotifier {
  TrainingSession? _currentSession;
  Timer? _pollingTimer;
  int _initialTokensTrained = 0;

  TrainingSession? get currentSession => _currentSession;
  int get initialTokensTrained => _initialTokensTrained;

  GlobalTrainingSessionProvider() {
    _startPolling();
  }

  void _startPolling() {
    _pollingTimer = Timer.periodic(
        const Duration(seconds: 5), (_) => refreshCurrentSession());
  }

  Future<void> refreshCurrentSession() async {
    try {
      final session = await API.getCurrentSession();
      if (_currentSession?.id != session?.id) {
        _currentSession = session;
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Error fetching current session: $e');
    }
  }

  void startSession(TrainingSession session) {
    _currentSession = session;
    _initialTokensTrained = session.tokensTrained;
    notifyListeners();
  }

  void updateInitialTokensTrained(int tokens) {
    _initialTokensTrained = tokens;
    notifyListeners();
  }

  void endSession() {
    _currentSession = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _pollingTimer?.cancel();
    super.dispose();
  }
}
