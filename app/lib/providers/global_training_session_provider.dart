import 'package:app/models/training_session.dart';
import 'package:flutter/foundation.dart';

class GlobalTrainingSessionProvider extends ChangeNotifier {
  bool _isSessionActive = false;
  TrainingSession? _currentSession;

  bool get isSessionActive => _isSessionActive;
  TrainingSession? get currentSession => _currentSession;

  void startSession(TrainingSession session) {
    _isSessionActive = true;
    _currentSession = session;
    notifyListeners();
  }

  void endSession() {
    _isSessionActive = false;
    _currentSession = null;
    notifyListeners();
  }
}
