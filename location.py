import time
import json
import os
from datetime import datetime
from typing import Optional, Dict, List


class TimezoneLocationDetector:
    """
    Privacy-friendly location detection using system timezone
    Fully autonomous version (no user input).
    """

    def __init__(self):
        self.timezone_mappings = self._build_timezone_mappings()
        self.dst_aware_mappings = self._build_dst_mappings()

    def _build_timezone_mappings(self) -> Dict[str, Dict]:
        """Comprehensive timezone to country mapping."""
        return {
            # North America
            'EST': {'country_code': 'us', 'country': 'United States', 'region': 'Eastern'},
            'CST': {'country_code': 'us', 'country': 'United States', 'region': 'Central'},
            'MST': {'country_code': 'us', 'country': 'United States', 'region': 'Mountain'},
            'PST': {'country_code': 'us', 'country': 'United States', 'region': 'Pacific'},
            'EDT': {'country_code': 'us', 'country': 'United States', 'region': 'Eastern'},
            'CDT': {'country_code': 'us', 'country': 'United States', 'region': 'Central'},
            'MDT': {'country_code': 'us', 'country': 'United States', 'region': 'Mountain'},
            'PDT': {'country_code': 'us', 'country': 'United States', 'region': 'Pacific'},
            'AKST': {'country_code': 'us', 'country': 'United States', 'region': 'Alaska'},
            'AKDT': {'country_code': 'us', 'country': 'United States', 'region': 'Alaska'},
            'HST': {'country_code': 'us', 'country': 'United States', 'region': 'Hawaii'},
            'AST': {'country_code': 'ca', 'country': 'Canada', 'region': 'Atlantic'},
            'ADT': {'country_code': 'ca', 'country': 'Canada', 'region': 'Atlantic'},

            # Europe
            'GMT': {'country_code': 'gb', 'country': 'United Kingdom', 'region': 'London'},
            'BST': {'country_code': 'gb', 'country': 'United Kingdom', 'region': 'London'},
            'CET': {'country_code': 'de', 'country': 'Germany', 'region': 'Central Europe'},
            'CEST': {'country_code': 'de', 'country': 'Germany', 'region': 'Central Europe'},
            'EET': {'country_code': 'fi', 'country': 'Finland', 'region': 'Eastern Europe'},
            'EEST': {'country_code': 'fi', 'country': 'Finland', 'region': 'Eastern Europe'},
            'WET': {'country_code': 'pt', 'country': 'Portugal', 'region': 'Western Europe'},
            'WEST': {'country_code': 'pt', 'country': 'Portugal', 'region': 'Western Europe'},

            # Asia Pacific
            'JST': {'country_code': 'jp', 'country': 'Japan', 'region': 'Tokyo'},
            'KST': {'country_code': 'kr', 'country': 'South Korea', 'region': 'Seoul'},
            'SGT': {'country_code': 'sg', 'country': 'Singapore', 'region': 'Singapore'},
            'IST': {'country_code': 'in', 'country': 'India', 'region': 'India'},
            'AEST': {'country_code': 'au', 'country': 'Australia', 'region': 'Eastern'},
            'AEDT': {'country_code': 'au', 'country': 'Australia', 'region': 'Eastern'},
            'AWST': {'country_code': 'au', 'country': 'Australia', 'region': 'Western'},
            'NZST': {'country_code': 'nz', 'country': 'New Zealand', 'region': 'Auckland'},
            'NZDT': {'country_code': 'nz', 'country': 'New Zealand', 'region': 'Auckland'},

            # Others
            'SAST': {'country_code': 'za', 'country': 'South Africa', 'region': 'Johannesburg'},
            'BRT': {'country_code': 'br', 'country': 'Brazil', 'region': 'Brasilia'},
            'ART': {'country_code': 'ar', 'country': 'Argentina', 'region': 'Buenos Aires'},
        }

    def _build_dst_mappings(self) -> Dict[int, List[str]]:
        """Map UTC offsets to possible countries."""
        return {
            -28800: ['us', 'ca'],  # UTC-8
            -25200: ['us', 'ca', 'mx'],  # UTC-7
            -21600: ['us', 'ca', 'mx'],  # UTC-6
            -18000: ['us', 'ca', 'mx'],  # UTC-5
            0: ['gb', 'ie', 'pt'],  # UTC+0
            3600: ['de', 'fr', 'it'],  # UTC+1
            7200: ['fi', 'ee', 'gr'],  # UTC+2
            10800: ['ru', 'tr', 'sa'],  # UTC+3
            19800: ['in', 'lk'],  # UTC+5:30
            28800: ['cn', 'sg', 'my'],  # UTC+8
            32400: ['jp', 'kr'],  # UTC+9
            36000: ['au', 'pg'],  # UTC+10
            43200: ['nz', 'fj'],  # UTC+12
        }

    def get_system_timezone_info(self) -> Dict[str, any]:
        """Extract timezone information from the system."""
        try:
            tz_abbrev = time.tzname[0] if not time.daylight else time.tzname[1]
            utc_offset = -time.timezone if not time.daylight else -time.altzone
            local_time = datetime.now()
            tz_env = os.environ.get('TZ', '')

            return {
                'abbreviation': tz_abbrev,
                'utc_offset_seconds': utc_offset,
                'utc_offset_hours': utc_offset / 3600,
                'local_time': local_time,
                'tz_environment': tz_env,
                'dst_names': time.tzname,
            }
        except Exception:
            return {}

    def detect_location_from_timezone(self) -> Optional[Dict]:
        """Detect location automatically."""
        tz_info = self.get_system_timezone_info()
        if not tz_info:
            return None

        tz_abbrev = tz_info.get('abbreviation', '')
        if tz_abbrev in self.timezone_mappings:
            location = self.timezone_mappings[tz_abbrev].copy()
            location.update({'timezone_info': tz_info})
            return location

        utc_offset = tz_info.get('utc_offset_seconds')
        if utc_offset is not None and utc_offset in self.dst_aware_mappings:
            primary_country = self.dst_aware_mappings[utc_offset][0]
            return {
                'country_code': primary_country,
                'country': primary_country.upper(),
                'region': '',
                'timezone_info': tz_info,
            }

        return None

    def get_location_auto(self) -> str:
        """Autonomous detection with caching."""
        location_data = self.detect_location_from_timezone()
        if not location_data:
            location_data = {
                'country_code': 'us',
                'country': 'United States',
                'region': '',
                'confidence': 'fallback'
            }
        return location_data.get('country_code', 'us')

def get_location() -> str:
    detector = TimezoneLocationDetector()
    return detector.get_location_auto()


if __name__ == "__main__":
    country_code = get_location()
    print(f"Detected country code: {country_code}")
