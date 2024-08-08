import 'package:app/models/training_session.dart';
import 'package:flutter/foundation.dart';
import 'package:app/api/api.dart';

class NewCorpusEntriesProvider extends ChangeNotifier {
  int _count = 0;
  bool _isLoading = false;

  int get count => _count;
  bool get isLoading => _isLoading;

  Future<void> fetchNewCorpusEntriesCount(TrainingSession session) async {
    _isLoading = true;
    notifyListeners();

    try {
      _count = await API.getNewCorpusEntriesCount(session.id);
    } catch (e) {
      debugPrint('Error fetching new corpus entries count: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
