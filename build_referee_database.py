#!/usr/bin/env python3
"""
build_referee_database.py - Costruisce il database statistiche arbitri

USO:
    python build_referee_database.py

COSTO API:
    ~10-15 chiamate totali (una per lega/stagione)

OUTPUT:
    referee_data.json con statistiche di tutti gli arbitri
"""

import sys
import os

# Import dal modulo
from fetch_referee_stats import build_referee_database, save_database
from config import API_FOOTBALL_KEY

if __name__ == "__main__":
    print("=" * 60)
    print("🏟️  COSTRUZIONE DATABASE STATISTICHE ARBITRI")
    print("=" * 60)
    print()
    
    # Verifica API key
    if not API_FOOTBALL_KEY:
        print("❌ ERRORE: API_FOOTBALL_KEY non configurata!")
        print("   Imposta la variabile d'ambiente o modifica config.py")
        sys.exit(1)
    
    print(f"✅ API Key configurata: {API_FOOTBALL_KEY[:8]}...")
    print()
    
    # Costruisci database
    database = build_referee_database()
    
    # Salva
    save_database(database)
    
    # Mostra statistiche finali
    print("\n" + "=" * 60)
    print("📊 STATISTICHE DATABASE")
    print("=" * 60)
    print(f"   Arbitri totali: {database.get('total_referees', 0)}")
    print(f"   Chiamate API usate: {database.get('api_calls_used', 0)}")
    print(f"   Stagioni analizzate: {database.get('seasons_analyzed', [])}")
    print(f"   Media globale cartellini: {database.get('global_average_cards', 0):.2f}")
    
    print("\n📁 Medie per lega:")
    for league, stats in database.get("league_averages", {}).items():
        print(f"   - {league}: {stats['avg_cards']:.2f} cart/partita ({stats['total_matches_analyzed']} partite)")
    
    # Top 5 più severi
    print("\n🔴 TOP 5 ARBITRI PIÙ SEVERI:")
    referees = database.get("referees", {})
    sorted_refs = sorted(referees.items(), key=lambda x: x[1]["severity_factor"], reverse=True)
    for i, (name, stats) in enumerate(sorted_refs[:5], 1):
        print(f"   {i}. {name} - {stats['avg_cards_per_match']:.1f} cart/partita (severity: {stats['severity_factor']:.2f})")
    
    # Top 5 più permissivi
    print("\n🟢 TOP 5 ARBITRI PIÙ PERMISSIVI:")
    for i, (name, stats) in enumerate(sorted_refs[-5:][::-1], 1):
        print(f"   {i}. {name} - {stats['avg_cards_per_match']:.1f} cart/partita (severity: {stats['severity_factor']:.2f})")
    
    print("\n" + "=" * 60)
    print("✅ DATABASE COSTRUITO CON SUCCESSO!")
    print("   File: referee_data.json")
    print("=" * 60)
