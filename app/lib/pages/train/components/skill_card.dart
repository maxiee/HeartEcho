import 'package:flutter/material.dart';

class SkillCard extends StatelessWidget {
  final String title;
  final String description;
  final VoidCallback onActivate;
  final bool isActive;

  const SkillCard({
    super.key,
    required this.title,
    required this.description,
    required this.onActivate,
    this.isActive = false,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Container(
        width: 250,
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(title, style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 8),
            SizedBox(height: 48, child: Text(description)),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: isActive ? null : onActivate,
              child: Text(isActive ? 'In Progress' : 'Activate'),
            ),
          ],
        ),
      ),
    );
  }
}
