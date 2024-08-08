import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:app/global_provider.dart';
import 'package:app/api/api.dart';
import 'package:app/models/training_session.dart';

class GlobalTitlebar extends StatefulWidget {
  const GlobalTitlebar({super.key});

  @override
  State createState() => _GlobalTitlebarState();
}

class _GlobalTitlebarState extends State<GlobalTitlebar> {
  Timer? _timer;
  TrainingSession? _currentSession;

  @override
  void initState() {
    super.initState();
    _startPolling();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _startPolling() {
    _timer = Timer.periodic(
        const Duration(seconds: 5), (_) => _fetchCurrentSession());
  }

  Future<void> _fetchCurrentSession() async {
    try {
      final session = await API.getCurrentSession();
      setState(() {
        _currentSession = session;
      });
    } catch (e) {
      // Handle error (e.g., log it or show a snackbar)
      debugPrint('Error fetching current session: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final globalProvider = Provider.of<GlobalProvider>(context);
    final currentMode = globalProvider.mode;

    return Container(
      height: 60,
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.5),
            spreadRadius: 1,
            blurRadius: 3,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 16.0),
            child: Text(
              "HeartEcho",
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.deepPurple,
              ),
            ),
          ),
          const Spacer(),
          _buildNavigationButtons(currentMode, globalProvider),
          const Spacer(),
          _buildSessionInfo(),
        ],
      ),
    );
  }

  Widget _buildNavigationButtons(
      Mode currentMode, GlobalProvider globalProvider) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _buildNavButton("聊天室", Mode.Chat, currentMode, globalProvider),
        const SizedBox(width: 4),
        _buildNavButton("语料库", Mode.Corpus, currentMode, globalProvider),
        const SizedBox(width: 4),
        _buildNavButton("炼丹炉", Mode.Train, currentMode, globalProvider),
      ],
    );
  }

  Widget _buildNavButton(String title, Mode mode, Mode currentMode,
      GlobalProvider globalProvider) {
    final isSelected = currentMode == mode;
    return InkWell(
      onTap: () {
        if (currentMode != mode) {
          globalProvider.changeMode(mode);
        }
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? Colors.deepPurple : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          title,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.black54,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }

  Widget _buildSessionInfo() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: _currentSession != null
          ? Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  'Session: ${_currentSession!.name}',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                Text(
                  'Last trained: ${_formatDateTime(_currentSession!.lastTrained)}',
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                ),
              ],
            )
          : const Text('No active session'),
    );
  }

  String _formatDateTime(DateTime dateTime) {
    return '${dateTime.year}-${dateTime.month.toString().padLeft(2, '0')}-${dateTime.day.toString().padLeft(2, '0')} '
        '${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';
  }
}
