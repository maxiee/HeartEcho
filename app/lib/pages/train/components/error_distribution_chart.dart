import 'package:app/models/corpus.dart';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:app/api/api.dart';

class ErrorDistributionChart extends StatefulWidget {
  final int refreshTrigger;

  const ErrorDistributionChart({super.key, required this.refreshTrigger});

  @override
  State<ErrorDistributionChart> createState() => _ErrorDistributionChartState();
}

class _ErrorDistributionChartState extends State<ErrorDistributionChart> {
  List<BarChartGroupData> _chartData = [];
  bool _isLoading = true;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _fetchData();
  }

  @override
  void didUpdateWidget(ErrorDistributionChart oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.refreshTrigger != oldWidget.refreshTrigger) {
      _fetchData();
    }
  }

  Future<void> _fetchData() async {
    try {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      final response = await API.getErrorDistribution();
      setState(() {
        _chartData = _createChartData(response.distribution);
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Error fetching error distribution: $e';
      });
    }
  }

  List<BarChartGroupData> _createChartData(
      List<LossDistributionItem> distribution) {
    return List.generate(distribution.length, (index) {
      final item = distribution[index];
      return BarChartGroupData(
        x: index,
        barRods: [
          BarChartRodData(
            toY: item.count.toDouble(),
            color: Colors.blue,
          ),
        ],
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_errorMessage.isNotEmpty) {
      return Center(child: Text(_errorMessage));
    }

    return Column(
      children: [
        Text('Error Distribution',
            style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 20),
        SizedBox(
          height: 300,
          child: BarChart(
            BarChartData(
              alignment: BarChartAlignment.spaceAround,
              maxY: _chartData.isEmpty ? 10 : null,
              barGroups: _chartData,
              titlesData: FlTitlesData(
                show: true,
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    getTitlesWidget: (value, meta) {
                      int index = value.toInt();
                      if (index % 2 == 0 && index < _chartData.length) {
                        return Text('${index * 0.5}-${(index + 1) * 0.5}');
                      }
                      return const Text('');
                    },
                  ),
                ),
                leftTitles:
                    const AxisTitles(sideTitles: SideTitles(showTitles: true)),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
